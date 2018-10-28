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