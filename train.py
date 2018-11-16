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
#from utils import *


class Trainer:
    def __init__(self, shape, mode, mean, std):
        self.mode = mode

        self.loader = get_loader(f'./data/{shape}/images',
                                 f'./data/{shape}/target.txt',
                                 selected_attrs=None, image_size=128,
                                 batch_size=64, dataset='Geometric',
                                 mode='train', num_workers=4, pin_memory=True,
                                 mean=[mean]*3, std=[std]*3)
        self.rot_loader = get_loader(f'./data/{shape}/no_translation',
                                     f'./data/{shape}/target.txt',
                                     selected_attrs=None, image_size=128,
                                     batch_size=64, dataset='Geometric',
                                     mode='train', num_workers=4,
                                     pin_memory=True,
                                     mean=[mean]*3, std=[std]*3)
        self.trans_loader = get_loader(f'./data/{shape}/no_rotation',
                                       f'./data/{shape}/target.txt',
                                       selected_attrs=None, image_size=128,
                                       batch_size=64, dataset='Geometric',
                                       mode='train', num_workers=4,
                                       pin_memory=True,
                                       mean=[mean]*3, std=[std]*3)
        self.loader_test = get_loader(f'./data/{shape}/images',
                                      f'./data/{shape}/target.txt',
                                      selected_attrs=None, image_size=128,
                                      batch_size=64, dataset='Geometric',
                                      mode='test', num_workers=4,
                                      pin_memory=True,
                                      mean=[mean]*3, std=[std]*3)
        self.rot_loader_test = get_loader(f'./data/{shape}/no_translation',
                                          f'./data/{shape}/target.txt',
                                          selected_attrs=None, image_size=128,
                                          batch_size=64, dataset='Geometric',
                                          mode='test', num_workers=4,
                                          pin_memory=True,
                                          mean=[mean]*3, std=[std]*3)
        self.rot_loader_test = get_loader(f'./data/{shape}/no_rotation',
                                          f'./data/{shape}/target.txt',
                                          selected_attrs=None, image_size=128,
                                          batch_size=64, dataset='Geometric',
                                          mode='test', num_workers=4,
                                          pin_memory=True,
                                          mean=[mean]*3, std=[std]*3)

        self.mean = mean
        self.std = std
        self.f_epoch = 1 / len(self.loader)
        self.f_eval = 1 / len(self.loader_test)
        self.writer = SummaryWriter()

    def train(self, model, epochs, optimizer, scheduler, loss_mod, device,
              save_dir=None):
        model = model.to(device)

        self.global_step = 0

        for epoch in range(1, epochs+1):
            model.train()
            print('_______________')
            losses = ae_epoch(model, loss_mod, optimizer, scheduler,
                              self.loader, self.rot_loader, device,
                              self.writer, self)

            losses = self.f_epoch * losses
            print(
                    'Epoch: {} \n'
                    'Training: \n'
                    'Loss: {:.3f} \t L_rec: {:.3f} \t'
                    'L_sparse:{:.3f} \t L_distr: {:.3f}'.format(
                        epoch, losses[0], losses[1], losses[2], losses[3])
            )

            model.eval()

            losses_test = ae_eval(model, loss_mod, self.loader_test,
                                  self.rot_loader_test, device, self.writer,
                                  self)

            losses_test = self.f_eval * losses_test
            self.writer.add_scalar('test/loss', losses_test[0],
                                   self.global_step)
            print(
                    '\n'
                    'Test: \n'
                    'Loss: {:.3f} \t L_rec: {:.3f} \t'
                    'L_sparse:{:.3f} \t L_distr: {:.3f}'.format(
                        losses_test[0], losses_test[1], losses_test[2],
                        losses_test[3]
                    )
            )

        return model


def ae_epoch(model, loss_mod, optimizer_gen, scheduler_gen, loader,
             eval_loader, device, writer, trainer):
    losses = np.zeros(5, dtype=np.double)
    scheduler_gen.step()
    for i, ((x1, label1), (x2, label2)) in tqdm(enumerate(zip(loader,
                                                              eval_loader))):
        x1 = x1.to(device)
        x2 = x2.to(device)
        z, x_ = model(x1)

        with torch.no_grad():
            x1 = x1 * trainer.std - trainer.mean
            x2 = x2 * trainer.std - trainer.mean

        loss = loss_mod([x2], x_, z)

        optimizer_gen.zero_grad()
        (loss[0]).backward(retain_graph=True)
        optimizer_gen.step()

        for j in range(len(loss)):
            losses[j] += loss[j].item() * 100

        writer.add_scalar('train/total_loss', losses[0], trainer.global_step)
        writer.add_scalar('train/loss_rec_rotation', losses[1], trainer.global_step)
        writer.add_scalar('train/loss_z_rotation', losses[2], trainer.global_step)
        writer.add_scalar('train/loss_rec_translation', losses[3], trainer.global_step)
        writer.add_scalar('train/loss_z_translation', losses[4], trainer.global_step)
        trainer.global_step += 1

    return losses


def ae_eval(model, loss_mod, loader, eval_loader, device, writer, trainer):
    losses = np.zeros(5, dtype=np.double)
    all_z = []
    all_angles = []
    plot_sample = True
    for (x1, label1), (x2, label2) in tqdm(zip(loader, eval_loader)):
        with torch.no_grad():
            x1 = x1.to(device)
            x2 = x2.to(device)
            z, x_ = model(x1)

            x1 = x1 * trainer.std - trainer.mean
            x2 = x2 * trainer.std - trainer.mean
            if plot_sample:
                plot_sample = False
                print(x1.mean(), x1.min(), x1.max(), x1.shape)
                writer.add_image('test/input',
                                 x1.cpu(),
                                 trainer.global_step)
                writer.add_image('test/target',
                                 x2.cpu(),
                                 trainer.global_step)
                writer.add_image('test/output',
                                 x_[0].cpu(),
                                 trainer.global_step)

            all_z.append(z[0])
            all_angles.append(label2[:, 3:])

            loss = loss_mod([x2], x_, z)
            # loss = [torch.nn.functional.mse_loss(x2, x_[0])]

            for i in range(len(loss)):
                losses[i] += loss[i].item() * 100

    all_z = torch.cat(all_z, dim=0)
    all_angles = torch.cat(all_angles, dim=0)

    for i in range(model.split):
        for j, name in enumerate(['theta', 'phi']):
            fig, ax = plt.subplots()
            ax.scatter(all_angles[:, j], all_z[:, i], s=2)
            ax.set(xlabel=f'$\\{name}$', ylabel='z')
            writer.add_figure(f'test/z{i}_{name}', fig, trainer.global_step)

    return losses


if __name__ == "__main__":
    model = Model(split=3, w=128)
    optimizer = torch.optim.Adam(model.parameters(), 0.0001)
    sched = torch.optim.lr_scheduler.StepLR(optimizer, 30, 0.1)
    trainer = Trainer(None, None, 0, 1)

    loss_module = Loss_Module(bootstrap_L2, lat_rot_loss)
    trainer.train(model, 100, optimizer, sched, loss_module, 'cuda')
