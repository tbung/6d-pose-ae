import torch
import torch.nn as nn
import torch.nn.functional as F


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

def weight_init(model, mean, std):
        for m in model._modules:
            normal_init(model._modules[m], mean, std)


class Encoder(nn.Module):
    #initializers
    def __init__(self,z_dim = 2 ,chan = 1,w = 64,d = 64):
        super(Encoder, self).__init__()

        k           = 3
        pad         = 0 
        st          = 2
        self.conv1  = nn.Conv2d(chan, d, k, stride=st, padding=pad )
        self.conv2  = nn.Conv2d(d, d * 2, k, stride=st, padding=pad)
        self.conv3  = nn.Conv2d(d*2, d*2, k, st, pad)
        self.conv4  = nn.Conv2d(d*2, d*4, k, st, pad)

        conv_size   = w
        for i in range (0, 4):
            conv_size =  int((conv_size - k +pad)/st +1)
        self.after_conv = conv_size
        self.size   = d*4 * self.after_conv **2
        self.fc     = nn.Linear(self.size, z_dim, bias = False)
        self.z_dim  = z_dim  


    # forward method
    def forward(self, input):
        x = F.relu((self.conv1(input)))
        x = F.relu((self.conv2(x)))
        x = F.relu((self.conv3(x)))
        x = F.relu((self.conv4(x)))
        x = x.view(-1, self.size)

        x = self.fc(x)

        return x



class Decoder(nn.Module):
    # initializers
    def __init__(self, z_dim = 2,chan = 1,w = 64,d = 64):
        super(Decoder, self).__init__()
        factor_ups  = 2**4
        fc_out      = w /factor_ups
        assert(fc_out%1 == 0) #width cannot be scaled up


        self.fc_out = int(fc_out)
        self.d      = d

        self.fc     = nn.Linear(z_dim, d*4*self.fc_out**2)
        

        k           = 3
        pad         = 1 
        st          = 1


        self.conv1  = nn.Conv2d(d*4, d*2, k, st, pad)
        self.conv2  = nn.Conv2d(d*2, d*2, k, st, pad)
        self.conv3  = nn.Conv2d(d*2, d, k, st, pad)
        self.conv4  = nn.Conv2d(d, chan, k, st, pad)

    # forward method
    def forward(self, input):
        # x = F.relu(self.deconv1(input))
        x = self.fc(input)
        x = x.view(-1, self.d*4, self.fc_out, self.fc_out)
        x = F.interpolate(x, scale_factor= 2)

        x = F.relu((self.conv1(x)))
        x = F.interpolate(x, scale_factor= 2)

        x = F.relu((self.conv2(x)))
        x = F.interpolate(x, scale_factor= 2)

        x = F.relu((self.conv3(x)))
        x = F.interpolate(x, scale_factor= 2)

        x = torch.tanh(self.conv4(x))

        return x

class Model(nn.Module):
    # initializers
    def __init__(self, split = 2 ,z_dim = 4,chan = 3,w = 64,d = 64):
        super(Model, self).__init__()
        self.encoder = Encoder(z_dim=z_dim, chan=chan, w=w, d=d)

        self.dec1    = Decoder(z_dim=split, chan=chan, w=w, d=d) 
        self.dec2    = Decoder(z_dim=z_dim-split, chan=chan, w=w, d=d)
        self.split   = split

    # forward method
    def forward(self, x, mode = 'no_trans'):
        """ 3 Different modes of forward:
        1. no_trans -> z1 encodes rotation      output: z1, x_rot
        2. no_rot   -> z2 encoder translation   output: z2, x_trans
        3. both     -> 1. & 2. combined         output: [z1,z2], [x_rot, x_trans]"""
        z   = self.encoder(x)
        z1  = z[:,:self.split].contiguous()
        z2  = z[:,self.split:].contiguous()
        
        if mode == 'no_trans':
            return z1, self.dec1(z1)
        
        elif mode == 'no_rot':
            return z2, self.dec2(z2)

        else:
            return [z1, z2], [self.dec1(z1), self.dec2(z2)]


def main():
    print("Test of single modules Encoder & Decoder")
    test_batch  = torch.randn(6, 3, 64, 64)
    print("shape of input for encoder")
    print(test_batch.shape)

    encoder     = Encoder(chan = 3)
    test_z      = encoder(test_batch)
    print("shape of the z")
    print(test_z.shape)
    decoder     = Decoder(chan = 3)
    test_rec    = decoder(test_z)
    print("shape of output of the decoder")
    print(test_rec.shape)

    print("\n \n Test of the combined model")
    model       = Model(chan = 3)
    z_, x_      = model(test_batch, mode='both')
    print('Shapes of z1 & z2')
    print(z_[0].shape, z_[1].shape)
    print('Shapes of x1 & x2')
    print(x_[0].shape, x_[1].shape)

if __name__ == "__main__":
    main()


