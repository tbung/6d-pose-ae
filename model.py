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


class Upsample(nn.Module):
    def __init__(self, factor=2):
        super(Upsample, self).__init__()
        self.factor = factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.factor)


class Encoder(nn.Module):
    # initializers
    def __init__(self, z_dim=2, chan=1, w=64, d=64, leaky=0, batch_norm=False):
        super(Encoder, self).__init__()

        k = 3
        pad = 0
        st = 2
        dims = [chan, d, d*2, d*4, d*4]

        if leaky > 0:
            self.activ = nn.LeakyReLU()
        else:
            self.activ = nn.ReLU()

        module_list = []
        for i in range(len(dims)-1):
            module_list.append(
                nn.Conv2d(dims[i], dims[i+1], k, stride=st, padding=pad, bias=not batch_norm))
            module_list.append(self.activ)
            if batch_norm is True:
                module_list.append(nn.BatchNorm2d(dims[i+1]))

        self.seq = nn.Sequential(*module_list)

        conv_size = w
        for i in range(0, 4):
            conv_size = int((conv_size - k + pad)/st + 1)
        self.after_conv = conv_size
        self.size = d*4 * self.after_conv ** 2
        self.fc = nn.Linear(self.size, z_dim, bias=False)
        self.z_dim = z_dim

    # forward method

    def forward(self, input):
        x = self.seq(input)

        x = x.view(-1, self.size)

        x = self.fc(x)

        return x


class Decoder(nn.Module):
    # initializers
    def __init__(self, z_dim=2, chan=1, w=64, d=64, leaky=0, batch_norm=False):
        super(Decoder, self).__init__()
        factor_ups = 2**4
        fc_out = w / factor_ups
        assert(fc_out % 1 == 0)  # width cannot be scaled up

        self.fc_out = int(fc_out)
        self.d = d

        self.fc = nn.Linear(z_dim, d*4*self.fc_out**2)

        k = 3
        pad = 1
        st = 1
        dims = [d*4, d*4, d*2, d, chan]

        if leaky > 0:
            self.activ = nn.LeakyReLU()
        else:
            self.activ = nn.ReLU()

        module_list = []
        for i in range(len(dims)-2):
            module_list.append(
                nn.Conv2d(dims[i], dims[i+1], k, stride=st, padding=pad, bias=not batch_norm))
            module_list.append(self.activ)
            if batch_norm is True:
                module_list.append(nn.BatchNorm2d(dims[i+1]))
            module_list.append(Upsample())
        module_list.append(
            nn.Conv2d(dims[-2], dims[-1], k, stride=st, padding=pad, bias=True))
        module_list.append(Upsample())

        self.seq = nn.Sequential(*module_list)

    # forward method
    def forward(self, input):

        x = self.fc(input)
        x = x.view(-1, self.d*4, self.fc_out, self.fc_out)
        x = self.seq(x)

        x = torch.sigmoid(x)

        return x


class Model(nn.Module):
    # initializers
    def __init__(self, trans_dim=3, rot_dim=4, chan=3, w=64, d=64, leaky=0, batch_norm=False):
        super(Model, self).__init__()
        z_dim = trans_dim + rot_dim
        split = rot_dim
        self.trans_dim = trans_dim
        self.rot_dim = rot_dim
        self.encoder = Encoder(z_dim=z_dim, chan=chan,
                               w=w, d=d, leaky=leaky, batch_norm=batch_norm)

        self.dec1 = Decoder(z_dim=split, chan=chan, w=w, d=d,
                            leaky=leaky, batch_norm=batch_norm)
        self.dec2 = Decoder(z_dim=z_dim-split, chan=chan,
                            w=w, d=d, leaky=leaky, batch_norm=batch_norm)
        self.split = split
        self.z_dim = z_dim

    # forward method
    def forward(self, x, mode='no_trans'):
        """ 3 Different modes of forward:
        1. no_trans -> z1 encodes rotation      output: [z1, z2], [x_rot, False]
        2. no_rot   -> z2 encoder translation   output: [z1, z2], [False, x_trans]
        3. both     -> 1. & 2. combined         output: [z1,z2], [x_rot, x_trans]"""
        z = self.encoder(x)
        z1 = z[:, :self.rot_dim].contiguous()
        z2 = z[:, self.rot_dim:].contiguous()

        if mode == 'no_trans':
            return [z1, z2], [self.dec1(z1), False]

        elif mode == 'no_rot':
            return [z1, z2], [False, self.dec2(z2)]

        else:
            return [z1, z2], [self.dec1(z1), self.dec2(z2)]


def main():
    print("Test of single modules Encoder & Decoder")
    test_batch = torch.randn(6, 3, 64, 64)
    print("shape of input for encoder")
    print(test_batch.shape)

    encoder = Encoder(chan=3)
    test_z = encoder(test_batch)
    print("shape of the z")
    print(test_z.shape)
    decoder = Decoder(chan=3)
    test_rec = decoder(test_z)
    print("shape of output of the decoder")
    print(test_rec.shape)

    print("\n \n Test of the combined model")
    model = Model(chan=3, batch_norm=True, leaky=0.1)
    z_, x_ = model(test_batch, mode='both')
    print('Shapes of z1 & z2')
    print(z_[0].shape, z_[1].shape)
    print('Shapes of x1 & x2')
    print(x_[0].shape, x_[1].shape)


if __name__ == "__main__":
    main()
