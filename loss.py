import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models

# all important loss module in which the losses are utilized
class Loss_Module(nn.Module):
    # initializer
    def __init__(self, loss_rec, loss_lat = None):
        super(Loss_Module, self).__init__()
        self.l_rec      = loss_rec
        self.l_lat   = loss_lat

    # forward pass
    def forward(self, x, x_ , z):
        l = x[0].new_zeros(5)
        v = 0
        for (xr, x_r, zr) in zip (x, x_, z):
            v += 1
            l[v] += self.l_rec(x_r, xr)

            if self.l_lat is not None:
                v += 1
                l[v] += self.l_lat(zr)

        l[0] += l.sum()

        return l


# reconstruction loss alternative to bootstrapped L2
def weighted_L2(x, y):
    l2  =  (x-y)**2
    b   = l2.size(0)
    l2  = l2.view(b, -1)
    mask = F.softmax(l2, dim = 1)
    wl2 = mask * l2 
    return wl2.sum()/b

# reconstruction loss alternative to bootstrapped L1
def weighted_L1(x, y):
    l1  =  torch.abs(x-y)
    b   = l1.size(0)
    l1  = l1.view(b, -1)
    mask = F.softmax(l1, dim = 1)
    wl2 = mask * l1 
    return wl2.sum()/b

# rot latent loss forcing the norm of z to 1
def lat_rot_loss(z):
    l   = (torch.abs(torch.norm(z, p=2, dim=1)-1.)).mean()
    return l


# bootstrapL2 as implemented in the paper
def bootstrap_L2(x, y, bootstrap_ratio=4):
    x_flat = x.reshape(x.shape[0], -1)
    y_flat = y.reshape(y.shape[0], -1)
    l2 = (x_flat - y_flat)**2
    l2, _ = torch.topk(l2, k=l2.shape[1]//bootstrap_ratio)
    return l2.sum()/(l2.shape[1]//bootstrap_ratio)

############################ VGG-Loss #####################################
# Normalization to VGG-normalization
class Norm_Image(nn.Module):

    def __init__(self, mean_in=[0, 0, 0], std_in=[1, 1, 1]):
        super(Norm_Image, self).__init__()
        
        self.fac  =  torch.tensor([0.229, 0.224, 0.225])
        self.mean = (torch.tensor(mean_in) - torch.tensor([0.485, 0.456, 0.406])) / self.fac
        self.fac  =  torch.tensor(std_in) / self.fac
        
        self.mean = nn.Parameter(self.mean.view(1, 3, 1, 1), requires_grad = False)
        self.fac  = nn.Parameter(self.fac.view(1, 3, 1, 1), requires_grad = False)

    def forward(self, inputs):
        return inputs*self.fac + self.mean

# Vgg-Features Extractor
class Vgg_Feat(nn.Module):
    def __init__(self, mean_in=[0.5]*3, std_in = [1]*3,requires_grad=False):
        super(Vgg_Feat, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        self.norm   = Norm_Image(mean_in=mean_in, std_in=std_in)

    def forward(self, X):
        h = self.norm(X)
        h = self.slice1(h)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        return [h_relu1_2, h_relu2_2]

# Vgg loss Module
class Vgg_Loss(nn.Module):
    def __init__(self, loss_mod,mean_in=[0.5]*3, std_in = [1]*3):
        super(Vgg_Loss, self).__init__()
        self.feat   = Vgg_Feat(mean_in=mean_in, std_in = std_in)

    def forward(self, x,y):
        x_feats = self.feat(x)
        y_feats = self.feat(y)

        loss = x.new_zeros(1)
        for x_feat,y_feat in zip(x_feats, y_feats):
            loss += loss_mod(x_feat, y_feat)

        return loss






######################### MAIN ############################
# test important parts of the modules
def main():
    print("Test of single modules Encoder & Decoder")
    zero_batch  = torch.zeros(6, 3, 64, 64)
    ones_batch  = torch.ones(6, 3, 64, 64)
    z1          = torch.randn(6,2)
    z0          = torch.zeros(6,2)
    loss_mod = Loss_Module(nn.MSELoss(), torch.norm )
    print("loss_0")
    loss_0      =  loss_mod([zero_batch,zero_batch], [zero_batch, ones_batch], [z0, z1])

    print(loss_0)


    print("\n \n Test weighted_L2 ")
    mask_test = torch.randn(4, 3, 3, 3)
    mask_0    = torch.zeros(4, 3, 3, 3)
    wl2       = weighted_L2(mask_test, mask_0)
    l2        = ((mask_test - mask_0)**2).mean()
    print(wl2)
    print(l2)

    print("\n \n Test lat_rot_loss")
    z0 = torch.randn(6,2)
    norm = torch.norm(z0, p=2, dim=1)
    print(norm.shape)
    z1  = z0/norm[:, None]
    print("z0 and corresponding loss")
    print (z0)
    print(lat_rot_loss(z0))
    print("normed to 1 z1 and corresponding loss")
    print(z1)
    print(lat_rot_loss(z1))

if __name__ == "__main__":
    main()
