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
                l[v] += self.l_lat(zr)

        l[0] += l.sum()

        return l


# reconstruction loss alternative to bootstrapped L2
def weighted_L2(x,y):
    l2  =  (x-y)**2
    b   = l2.size(0)
    l2  = l2.view(b, -1)
    mask = F.softmax(l2, dim = 1)
    wl2 = mask * l2 
    return wl2.sum()/b

# rot latent loss forcing the norm of z to 1
def lat_rot_loss(z):
    l   = (torch.abs(torch.norm(z, p=2, dim=1)-1.)).mean()
    return l



def bootstrap_L2(x, y, bootstrap_ratio=4):
    x_flat = x.reshape(x.shape[0], -1)
    y_flat = y.reshape(y.shape[0], -1)
    l2 = (x_flat - y_flat)**2
    l2, _ = torch.topk(l2, k=l2.shape[1]//bootstrap_ratio)
    return l2.mean()


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
