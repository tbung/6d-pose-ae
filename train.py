import torch
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from tensorboardX import SummaryWriter

from pathlib import Path
from loss import Loss_Module, bootstrap_L2, lat_rot_loss
from data_loader import get_loader
from model import Model


class Trainer:
    def __init__(self, shape, mode, mean, std):
        self.mode = mode
        self.loader = get_loader(f'./data/{shape}', image_size=128,
                                 batch_size=64, dataset='Geometric',
                                 mode='train', num_workers=4, pin_memory=True,
                                 mean=[mean]*3, std=[std]*3)

        self.loader_test = get_loader(f'./data/{shape}', image_size=128,
                                      batch_size=64, dataset='Geometric',
                                      mode='test', num_workers=4,
                                      pin_memory=True, mean=[mean]*3,
                                      std=[std]*3)

        self.mean = mean
        self.std = std
        self.f_epoch = 1 / len(self.loader)
        self.f_eval = 1 / len(self.loader_test)
        self.writer = SummaryWriter()

    def train(self, model, epochs, optimizer, scheduler, loss_mod, device,
              mode='no_trans'):
        """ 3 Different modes of Training the forward:
        1. no_trans -> z1 encodes rotation      output: z1, x_rot
        2. no_rot   -> z2 encoder translation   output: z2, x_trans
        3. both     -> 1. & 2. combined         output: [z1,z2], [x_rot, x_trans]"""

        model = model.to(device)

        self.global_step = 0

        for epoch in range(1, epochs+1):
            model.train()
            print('_______________')
            losses = ae_epoch(model, loss_mod, optimizer, scheduler,
                              self.loader, mode, device,
                              self.writer, self)

            losses = self.f_epoch * losses
            print(
                    'Epoch: {} \n'
                    'Training: \n'
                    'Loss: {:.3f} \t L_rot: {:.3f} \t'
                    'L_rot_z:{:.3f} \t L_trans: {:.3f} \t'
                    'L_trans: {:.3f}'.format(
                        epoch, losses[0], losses[1], losses[2], losses[3], losses[4])
            )

            model.eval()

            losses_test = ae_eval(model, loss_mod, self.loader_test, mode,
                                  device, self.writer, self)

            losses_test = self.f_eval * losses_test
            self.writer.add_scalar('test/loss', losses_test[0],
                                   self.global_step)
            print(
                    '\n'
                    'Test: \n'
                    'Loss: {:.3f} \t L_rot: {:.3f} \t'
                    'L_rot_z:{:.3f} \t L_trans: {:.3f} \t'
                    'L_trans: {:.3f}'.format(
                        losses_test[0], losses_test[1], losses_test[2],
                        losses_test[3], losses_test[4]
                    )
            )

        return model

    def normalize(self, x):
        return x * self.std - self.mean


def ae_epoch(model, loss_mod, optimizer_gen, scheduler_gen, loader,
             mode, device, writer, trainer):
    losses = np.zeros(5, dtype=np.double)
    scheduler_gen.step()
    for i, (x1, x2, x3, labels) in tqdm(enumerate(loader)):
        x1 = x1.to(device)
        x2 = x2.to(device)
        x3 = x3.to(device)
        z, x_ = model(x1)

        with torch.no_grad():
            x1 = trainer.normalize(x1)
            x2 = trainer.normalize(x2)
            x3 = trainer.normalize(x3)
        if mode == 'no_rot':
            val_x = [x3]
        elif mode == 'no_trans':
            val_x = [x2]
        else:
            val_x = [x2, x3]

        loss = loss_mod(val_x, x_, z, mode=mode)

        optimizer_gen.zero_grad()
        (loss[0]).backward(retain_graph=True)
        optimizer_gen.step()

        for j in range(len(loss)):
            losses[j] += loss[j].item() * 100

        writer.add_scalar('train/total_loss', losses[0], trainer.global_step)
        writer.add_scalar('train/loss_rec_rotation', losses[1],
                          trainer.global_step)
        writer.add_scalar('train/loss_z_rotation', losses[2],
                          trainer.global_step)
        writer.add_scalar('train/loss_rec_translation', losses[3],
                          trainer.global_step)
        writer.add_scalar('train/loss_z_translation', losses[4],
                          trainer.global_step)
        trainer.global_step += 1

    return losses


def ae_eval(model, loss_mod, loader, mode, device, writer, trainer):
    losses = np.zeros(5, dtype=np.double)
    all_z = []
    all_z_ax = []
    all_angles = []
    all_axis = []
    plot_sample = True
    for i, (x1, x2, x3, label) in tqdm(enumerate(loader)):
        with torch.no_grad():
            x1 = x1.to(device)
            x2 = x2.to(device)
            x3 = x3.to(device)
            z, x_ = model(x1)

            x1 = trainer.normalize(x1)
            x2 = trainer.normalize(x2)
            x3 = trainer.normalize(x3)

            if mode == 'no_rot':
                val_x = [x3]
            elif mode == 'no_trans':
                val_x = [x2]
            else:
                val_x = [x2, x3]

            if plot_sample:
                plot_sample = False
                print(x1.mean(), x1.min(), x1.max(), x1.shape)
                writer.add_image('test/input',
                                 x1.cpu(),
                                 trainer.global_step)
                writer.add_image('test/target',
                                 val_x[0].cpu(),
                                 trainer.global_step)
                writer.add_image('test/output',
                                 x_[0].cpu(),
                                 trainer.global_step)
                if mode == 'both':
                    writer.add_image('test/target_1',
                                     val_x[1].cpu(),
                                     trainer.global_step)
                    writer.add_image('test/output_1',
                                     x_[1].cpu(),
                                     trainer.global_step)

            all_z.append(z[0])
            all_angles.append(label[:, 3:])
            all_axis.append(label[:, :3])
            all_z_ax.append(z[1])

            loss = loss_mod(val_x, x_, z, mode=mode)

            for i in range(len(loss)):
                losses[i] += loss[i].item() * 100

    all_z = torch.cat(all_z, dim=0)
    all_z_ax = torch.cat(all_z_ax, dim=0)
    all_angles = torch.cat(all_angles, dim=0)
    all_axis = torch.cat(all_axis, dim=0)

    for i in range(model.split):

        if mode != 'no_rot':
            for j, name in enumerate(['theta', 'phi']):
                fig, ax = plt.subplots()
                ax.scatter(all_angles[:, j], all_z[:, i], s=2)
                ax.set(xlabel=f'$\\{name}$', ylabel='z')
                writer.add_figure(f'test/z{i}_{name}', fig,
                                  trainer.global_step)
        if mode != 'no_trans':
            for j, name in enumerate(['x', 'y', 'z']):
                fig, ax = plt.subplots()
                ax.scatter(all_axis[:, j], all_z_ax[:, i], s=2)
                ax.set(xlabel=f'$\\{name}$', ylabel='z')
                writer.add_figure(f'test/z{i}_{name}', fig,
                                  trainer.global_step)

    return losses


if __name__ == "__main__":
    model = Model(split=3, w=128)
    optimizer = torch.optim.Adam(model.parameters(), 0.0001)
    sched = torch.optim.lr_scheduler.StepLR(optimizer, 30, 0.1)
    trainer = Trainer('square', 'no_rot', 0, 1)

    loss_module = Loss_Module(bootstrap_L2, lat_rot_loss)
    trainer.train(model, 100, optimizer, sched, loss_module, 'cuda')
