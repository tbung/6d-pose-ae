import torch
import numpy as np
import os
import torch.nn.functional as F
import torch.nn as nn
from scipy.stats import mode

class tracker():
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
    print(path)


# Test the nearest_cosine function with indices
def get_nearest_cosine(z, z_book, label_book ,k, device):
    #z, z_book = z.to(device), z_book.to(device)
    with torch.no_grad():
        z_  = renorm(z)
        z_cos= (z_[:, None, :] * z_book[None, :, :]).sum(dim=2)
        vals, ind = z_cos.topk(k ,dim=1)
    
    labels  = label_book[ind]
    labels  = labels.to(device)

    return vals, ind, labels

# Get KNN for eucildean distance
def get_nearest_euclidean(z, z_book, label_book ,k, device):
    #z, z_book = z.to(device), z_book.to(device)
    with torch.no_grad():
        m2 = - ((z[:, None, :] - z_book[None, :, :])**2).sum(dim=2)
        vals, ind = m2.topk(k ,dim=1)
    
    labels  = label_book[ind]
    labels  = labels.to(device)

    return vals, ind, labels

def renorm(x):
    return x/x.norm(p=2, dim = 1)[:,None]


# Implementation of mean of KNN for regression problem
def lazy_mean(vals, ind, labels):
    return labels.mean(dim = 1)

def weighted_mean(vals, ind, labels):
    weights   = F.softmax(vals, dim=1)
    return (labels*weights[:,:,None]).mean(dim=1)


# Implementation of KNN for decision problem
def mode_knn(vals, ind, labels):
    device = labels.device
    labels = labels.to('cpu').numpy()
    #print( mode(labels, axis=1))
    return (torch.Tensor(mode(labels, axis=1)[0])).to(device)

# Creating the codebook with a set data_loader 
def create_codetensors(model, data_loader, device, step_ax = 0.1, step_rot = 1.):
    model = model.to(device)
    z_rot_book = []
    z_ax_book = []
    axis_book = []
    rot_book  = []
    for i, (x, label) in enumerate(data_loader):
        x   = x.to(device)
        z   = model.encoder(x)
        z1  = z[:,:model.split]
        z2  = z[:, model.split:]
        z1  = renorm(z1)
        #z2  = renorm(z2) # z2 will be assumed to be 4dim with norm also set to 1
        #z_rot_book.append(z1.to('cpu'))
        #z_ax_book.append(z2.to('cpu'))
        z_rot_book.append(z1)
        z_ax_book.append(z2)
        axis_book.append(label[:, :3]//step_ax * step_ax)
        rot_book.append(label[:,3:]//step_rot * step_rot)

    z_rot_book = torch.cat(z_rot_book, dim=0)
    z_ax_book = torch.cat(z_ax_book, dim=0)

    axis_book = torch.cat(axis_book, dim=0)
    rot_book = torch.cat(rot_book, dim=0)

    return z_rot_book, rot_book, z_ax_book, axis_book

class Codebook(nn.Module):
    def __init__(self,model, data_loader, device):
        super(Codebook, self).__init__()
        self.model = model
        self.register_buffer('z_rot', None)
        self.register_buffer('z_trans', None)
        self.register_buffer('rot', None)
        self.register_buffer('trans', None)

        self.init_book(data_loader, device)
        self = self.to(device)

    def init_book(self, data_loader, device):
        books =  create_codetensors(self.model, data_loader, device, step_ax = 0.1, step_rot = 1.)
        for book in books:
            book = book.to(device)
        self.z_rot.data = book[0]
        self.rot.data = book[1]
        self.z_trans.data = book[2]
        self.trans.data = book[3]
        print('finished codebook initalization')

        
    def forward(self, inputs):
        #z_list[0] -> z_rot
        #z_list[1] -> z_trans
        z_list, x_list = self.model(inputs)
        k = 20
        rotations = self.rot_module(get_nearest_cosine(z_list[0], self.z_rot, self.rot , k, self.device))
        translations = self.trans_module(get_nearest_euclidean(z_list[1], self.z_trans, self.trans , k, self.device) )


        return rotations, translations
        


    




def main():
    z = torch.ones(10,2)
    z = renorm(z)
    z_book = torch.randn(100,2)
    z_book = renorm(z_book)
    label_book = torch.stack([(torch.arange(0,100, dtype = torch.float)//2)]*2, dim = 1)
    print(label_book.shape)
    
    vals, ind, labels = get_nearest_cosine(z, z_book, label_book ,3, 'cpu')
    print('vals \n', vals)
    print('ind \n', ind)
    print('labels.shape \n', labels.shape)
    
    print('lazy mean \n', lazy_mean(vals, ind, labels))

    print('weighted mean', weighted_mean(vals, ind, labels))

    print('mode kNN', mode_knn(vals, ind, labels))

    



        
if __name__ == "__main__":
    main()