import torch
import numpy as np
import os
import torch.nn.functional as F
import torch.nn as nn
from scipy.stats import mode
import math

"""class tracker():
    def __init__(self):
        self.mu     = 0
        self.std    = 0
        self.count  = 0
        self.mu_dim = None
        self.std_dim= None

    def update(self, v):
        b           =   v.size(0)

        if self.mu_dim is None:
            self.mu_dim     = self.get_mu(v)
            self.std_dim    = self.get_std(v)

        else:
            self.mu_dim     = (b * self.get_mu(v) + self.count * self.mu)/(b + self.count)
            self.std_dim    =   ((b * (self.get_std(v))**2 + self.count * self.std**2)/(b + self.count))**0.5

        self.mu     =   self.mu_dim.mean().item()
        self.std    =   self.std_dim.mean().item()
        self.count  +=  b

    def get_mu(self, v):
        with torch.no_grad():
            mu_dim = torch.mean(v, dim = 0).to('cpu')
        return mu_dim


    def get_std(self, v):
        with torch.no_grad():
            std  = torch.std(v, dim = 0).to('cpu')
        return std
    
    def reset(self):
        self.mu     = 0
        self.std    = 0
        self.count  = 0
        self.mu_dim = None
        self.std_dim= None

def save_log(folder, array):
    path = os.path.join(folder, 'loss_log.npy')
    np.save(path, array)

def create_folder(folder):
    path = os.path.join(folder, 'images','test.pth')
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    print('Folder created.')
    print(path)"""


# Test the nearest_cosine function with indices
def get_nearest_cosine(z, z_book, label_book, k, device):
    #z, z_book = z.to(device), z_book.to(device)
    with torch.no_grad():
        z_ = renorm(z)
        z_cos = (z_[:, None, :] * z_book[None, :, :]).sum(dim=2)
        vals, ind = z_cos.topk(k, dim=1)

    labels = label_book[ind]
    labels = labels.to(device)

    return vals, ind, labels

# Get KNN for eucildean distance


def get_nearest_euclidean(z, z_book, label_book, k, device):
    #z, z_book = z.to(device), z_book.to(device)
    with torch.no_grad():
        m2 = - ((z[:, None, :] - z_book[None, :, :])**2).sum(dim=2)
        vals, ind = m2.topk(k, dim=1)

    labels = label_book[ind]
    labels = labels.to(device)

    return vals, ind, labels


def renorm(x):
    return x/x.norm(p=2, dim=1)[:, None]


# Implementation of mean of KNN for regression problem
def lazy_mean(vals, ind, labels):
    return labels.mean(dim=1)

# Implementation of weighted mean for regression problem


def weighted_mean(vals, ind, labels):
    weights = F.softmax(vals, dim=1)
    return (labels*weights[:, :, None]).mean(dim=1)


# Implementation of KNN for decision problem
def mode_knn(vals, ind, labels):
    device = labels.device
    labels = labels.to('cpu').numpy()
    #print( mode(labels, axis=1))
    return (torch.Tensor(mode(labels, axis=1)[0])).to(device)

# Creating the codebook with a set data_loader


def create_codetensors(model, data_loader, device, step_ax=0.1, step_rot=1.):
    model = model.to(device)
    z_rot_book = []
    z_ax_book = []
    axis_book = []
    rot_book = []
    with torch.no_grad():
        for i, (x, _1, _2, label) in enumerate(data_loader):
            x = x.to(device)
            z = model.encoder(x)
            z1 = z[:, :model.split]
            z2 = z[:, model.split:]
            z1 = renorm(z1)
            # z2  = renorm(z2) # z2 will be assumed to be 4dim with norm also set to 1
            # z_rot_book.append(z1.to('cpu'))
            # z_ax_book.append(z2.to('cpu'))
            z_rot_book.append(z1)
            z_ax_book.append(z2)
            axis_book.append(label[:, :3]//step_ax * step_ax)
            rot_book.append(label[:, 3:]//step_rot * step_rot)

        z_rot_book = torch.cat(z_rot_book, dim=0)
        z_ax_book = torch.cat(z_ax_book, dim=0)

        axis_book = torch.cat(axis_book, dim=0)
        rot_book = torch.cat(rot_book, dim=0)

    return z_rot_book, rot_book, z_ax_book, axis_book


class Codebook(nn.Module):
    def __init__(self, model, data_loader, device):
        super(Codebook, self).__init__()
        self.model = model
        self.step_ax = 0.1
        self.step_rot = 1
        model = model.eval()
        self.register_buffer('z_rot', None)
        self.register_buffer('z_trans', None)
        self.register_buffer('rot', None)
        self.register_buffer('trans', None)

        self.init_book(data_loader, device)
        self = self.to(device)
        self.k = 1

        # different versions to extract the information of the latent space
        self.rot_module = mode_knn
        self.trans_module = weighted_mean

    def init_book(self, data_loader, device):
        books = create_codetensors(
            self.model, data_loader, device, step_ax=self.step_ax, step_rot=self.step_rot)
        for book in books:
            book = book.to(device)
        self.z_rot = books[0]
        self.rot = books[1]
        self.z_trans = books[2]
        self.trans = books[3]
        print('finished codebook initalization')

    def forward(self, inputs):
        # z_list[0] -> z_rot
        # z_list[1] -> z_trans
        z_list, x_list = self.model(inputs)
        tulple1 = get_nearest_cosine(
            z_list[0], self.z_rot, self.rot, self.k, next(self.buffers()).device)
        tulple2 = get_nearest_euclidean(
            z_list[1], self.z_trans, self.trans, self.k, next(self.buffers()).device)
        rotations = self.rot_module(*tulple1)
        translations = self.trans_module(*tulple2)

        return rotations, translations

# Function taken from https://github.com/facebookresearch/QuaterNet/blob/master/common/quaternion.py


def qmul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)


############ Code Taken from https://www.learnopencv.com/rotation-matrix-to-euler-angles/ #################
# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    RtR = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - RtR)
    return n < 1e-6


# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta):

    R_x = np.array([[1,         0,                  0],
                    [0,         math.cos(theta[0]), -math.sin(theta[0])],
                    [0,         math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])],
                    [0,                     1,      0],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


def rotationMatrixToEulerAngles(R):

    assert(isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])
############################ END Taken Code ##########################

# gets 2 3dim vectors with euler vectors and outputs the mix of both
# receives numpy arrays as inputs and outputs numpy arrays


def get_euler(tulple1, tulple2):
    R1 = eulerAnglesToRotationMatrix(tulple1)
    R2 = eulerAnglesToRotationMatrix(tulple2)

    R3 = np.dot(R2, R1)
    return rotationMatrixToEulerAngles(R3)


# Base Code taken from: https://en.wikipedia.org/wiki/Slerp
def slerp(v0, v1, t_array):
    # input of 2 quaternions v0 & v1 as torch tensors
    # t_array goes from 0 to 1 and handles the interpolation steps e.g. torch.arange(0, 1, 0.1)
    #t_array = torch.tensor(t_array)
    dot = (v0*v1).sum()

    if (dot < 0.0):
        v1 = -v1
        dot = -dot

    DOT_THRESHOLD = 0.9995
    if (dot > DOT_THRESHOLD):
        result = v0[None, :] + t_array[:, None]*(v1 - v0)[None, :]
        result = renorm(result)
        return result

    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)

    theta = theta_0*t_array
    sin_theta = torch.sin(theta)

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return (s0[:, None] * v0[None, :]) + (s1[:, None] * v1[None, :])


def symmetries(label, object_type='square'):
    assert(object_type != 'eggbox')  # symmetries for eggbox not defined yet
    if object_type == 'cat':
        return label
    else:
        return torch.fmod(label, 90)


def main():
    z = torch.ones(10, 2)
    z = renorm(z)
    z_book = torch.randn(100, 2)
    z_book = renorm(z_book)
    label_book = torch.stack(
        [(torch.arange(0, 100, dtype=torch.float)//2)]*2, dim=1)
    print(label_book.shape)

    vals, ind, labels = get_nearest_cosine(z, z_book, label_book, 3, 'cpu')
    print('vals \n', vals)
    print('ind \n', ind)
    print('labels.shape \n', labels.shape)

    print('lazy mean \n', lazy_mean(vals, ind, labels))

    print('weighted mean', weighted_mean(vals, ind, labels))

    print('mode kNN', mode_knn(vals, ind, labels))
    z_slerp = slerp(z_book[0], z_book[1], torch.arange(0, 1, 0.3))
    print(z_slerp)


if __name__ == "__main__":
    main()
