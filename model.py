import torch
import torch.nn as nn
import torch.nn.functional as F


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


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

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

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


    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

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

        x = F.relu((self.conv4(x)))

        return x

def main():
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

if __name__ == "__main__":
    main()


