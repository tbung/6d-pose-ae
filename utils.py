import torch
import numpy as np
import os

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
def get_nearest_cosine(z, z_book):
    with torch.no_grad():
        z_  = z/(torch.norm(z, p=2, dim=1)[:, None])
        z_cos= (z_[:, None, :] * z_book[None, :, :]).sum(dim=2)
        vals, ind = z_cos.topk(20 ,dim=1)
    return vals, ind

# Creating the codebook with a set data_loader 
def create_codebook(model, train, data_loader, device):
    model = model.to(device)
    z1_book = []
    z2_book = []
    for i, (x, label) in enumerate(data_loader):
        x   = x.to(device)
        z   = model.encoder(x)
        z1  = z[:,:model.split]
        z2  = z[:, model.split:]
        z1  = z1/(torch.norm(z1, p=2, dim=1))
        z2  = z2/(torch.norm(z1, p=2, dim=1)) # z2 will be assumed to be 4dim with norm also set to 1
        z1_book.append(z1.to('cpu'))
        z2_book.append(z2.to('cpu'))

    z1_book = torch.stack(z1_book, dim=0)
    z2_book = torch.stack(z2_book, dim=0)

    return 

        
        
      