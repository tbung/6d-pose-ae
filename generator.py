import torch
import torch.nn as nn
import torch.nn.functional as F


class Upsample(nn.Module):
    def __init__(self, in_channels, k = 2):
        super(Upsample, self).__init__()
        self.register_buffer('weight', torch.ones((k,k)))
        self.channels = in_channels

    def forward(self, input):
        return F.torch.nn.functional.conv_transpose2d(input, self.weight, bias=None, stride=k, padding=0, output_padding=0, groups=self.channels)

class Decoder(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(self, Decoder).__init__()
        self.fc     = nn.Linear(100, 16*d*4)


        self.usa1   = Upsample(d*4, 2)
        self.conv1  = nn.Conv2d(d*4, d*2, 5, 1, 2)
        self.usa2   = Upsample(d*2, 2)
        self.conv2  = nn.Conv2d(d*2, d*2, 5, 1, 2)
        self.usa3   = Upsample(d*2, 2)
        self.conv3  = nn.Conv2d(d*2, d, 5, 1, 2)
        self.usa4   = Upsample(d, 2)
        self.conv4  = nn.Conv2d(d, 3, 5, 1, 2)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        # x = F.relu(self.deconv1(input))
        x = self.fc(input)
        x = x.view(-1, 512, 8, 8)
        x = F.relu((self.conv1(x)))
        x = F.relu((self.conv2(x)))
        x = F.relu((self.conv3(x)))
        x = F.relu((self.conv4(x)))
        x = torch.tanh(self.deconv5(x))

        return x

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class Encoder(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(self, Encoder).__init__()

        self.fc     = nn.Linear(16*d*4, 100)
        
        self.conv4 = nn.Conv2d(d*2, d*4, 5, 2, 0)
        self.conv3 = nn.Conv2d(d*2, d*2, 5, 2, 0)
        self.conv2 = nn.Conv2d(d, d*2, 5, 2, 0)
        self.conv1 = nn.Conv2d(3, d, 5, 2, 0)

        self.size      = d

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        # x = F.relu(self.deconv1(input))
        x = F.relu((self.conv1(input)))
        x = F.relu((self.conv2(x)))
        x = F.relu((self.conv3(x)))
        x = F.relu((self.conv4(x)))
        #x = x.view(-1, 8 *8 *self.d * 4)

        #x = self.fc(x)

        return x

class smallEncoder(nn.Module):
    #initializers
    def __init__(self, chan = 1,w = 64,d = 64):
        super(self, smallEncoder).__init__()
        self.fc     = nn.Linear(chan *100, 2, bias = False)
        self.conv2  = nn.Conv2d(d, d * 2, 5, stride=2, padding= 0)
        self.conv1  = nn.Conv2d(chan, d, 5, stride = 2, padding= 0 )

        self.size   = d
        
    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
        
    # forward method
    def forward(self, input):
        x = F.relu((self.conv1(input)))
        x = F.relu((self.conv2(x)))
        #x = x.view(-1, self.size)

        #x = self.fc(x)

        return x



class smallDecoder(nn.Module):
    # initializers
    def __init__(self, chan = 1,w = 64,d = 64):
        super(self, smallDecoder).__init__()
        self.fc     = nn.Linear(100, 16*d*4)


        self.usa1   = Upsample(d*4, 2)
        self.conv1  = nn.Conv2d(d*2, d, 5, 1, 2)
        self.usa2   = Upsample(d, 2)
        self.conv2  = nn.Conv2d(d, chan, 5, 1, 2)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        # x = F.relu(self.deconv1(input))
        x = self.fc(input)
        x = x.view(-1, 512, 8, 8)
        x = F.relu((self.conv1(x)))
        x = F.relu((self.conv2(x)))
        x = F.relu((self.conv3(x)))
        x = F.relu((self.conv4(x)))
        x = torch.tanh(self.deconv5(x))

        return x


