import torch.nn as nn
import torch.nn.functional as F
import torch


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
                l[v] += self.l_lat(zr) /zr.size(0)
        
        l[0] += l.sum()

        return l


def weighted_L2(x,y):
    l2  =  (x-y)**2
    b   = l2.size(0)
    l2  = l2.view(b, -1)
    mask = F.softmax(l2, dim = 1)
    wl2 = mask * l2 
    return wl2.sum()/b

def lat_rot_loss(z):
    




def main():
    print("Test of single modules Encoder & Decoder")
    zero_batch  = torch.zeros(6, 3, 64, 64)
    ones_batch  = torch.ones(6, 3, 64, 64)
    z1          = torch.randn(6,2)
    z0         = torch.zeros(6,2)
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



if __name__ == "__main__":
    main()
                