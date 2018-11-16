import torch
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from tensorboardX import SummaryWriter

from pathlib import Path

from data_loader import get_loader
from model import Model
from loss import Loss_Module, bootstrap_L2
from utils import *


class Trainer:
    def __init__(self, image_dir, attr_path):
        self.loader = get_loader('./data/square/images',
                                 './data/square/target.txt',
                                 selected_attrs=None, image_size=64,
                                 batch_size=64, dataset='Geometric',
                                 mode='train', num_workers=4, pin_memory=True)
        self.rot_loader = get_loader('./data/square/no_translation',
                                     './data/square/target.txt',
                                     selected_attrs=None, image_size=64,
                                     batch_size=64, dataset='Geometric',
                                     mode='train', num_workers=4,
                                     pin_memory=True)
        self.loader_test = get_loader('./data/square/images',
                                      './data/square/target.txt',
                                      selected_attrs=None, image_size=64,
                                      batch_size=64, dataset='Geometric',
                                      mode='test', num_workers=4,
                                      pin_memory=True)
        self.rot_loader_test = get_loader('./data/square/no_translation',
                                          './data/square/target.txt',
                                          selected_attrs=None, image_size=64,
                                          batch_size=64, dataset='Geometric',
                                          mode='test', num_workers=4,
                                          pin_memory=True)
        # self.dist_loader= get_loader(image_dir, attr_path, selected_attrs =
        # None, image_size=64,
        #       batch_size=64, dataset='Geometric', mode='train',
        #       num_workers=4, pin_memory = True)
        self.mean = 0.5
        self.std = 0.5
        self.f_epoch = 1 / len(self.loader)
        self.f_eval = 1 / len(self.loader_test)
        self.writer = SummaryWriter()

    def train(self, model, epochs, optimizer, scheduler, loss_mod, device,
              save_dir=None):
        model = model.to(device)
        # loss_mod = loss_mod.to(device)

        self.global_step = 0

        if save_dir is not None:
            loss_array = []
            path_im = Path('./images', 'ae', save_dir)
            path_run = Path('./saved_models', 'ae', save_dir)
            path_im.mkdir(parents=True, exist_ok=True)
            path_run.mkdir(parents=True, exist_ok=True)

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
            self.writer.add_scalar('test/loss', losses_test[0], self.global_step)
            print(
                    '\n'
                    'Test: \n'
                    'Loss: {:.3f} \t L_rec: {:.3f} \t'
                    'L_sparse:{:.3f} \t L_distr: {:.3f}'.format(
                        losses_test[0], losses_test[1], losses_test[2],
                        losses_test[3]
                    )
            )

            if save_dir is not None:
                with torch.no_grad():
                    loss_array.append(np.append(losses, losses_test))

                    x = next(iter(self.loader))[0].to(device)
                    _, x_ = model(x)
                    x_ = x_ * self.std + self.mean
                    x_rec = x_[:, :3].clamp(0, 1).to('cpu')
                    vutils.save_image(x_rec,
                                      path_im / f'{epoch}_epoch_rec.png')
                    torch.save(model.state_dict(),
                               path_run / f'{epoch}_model.pth')

        if save_dir is not None:
            loss_array = np.stack(loss_array, axis=0)
            save_log(path_im, loss_array)
        return model


def ae_epoch(model, loss_mod, optimizer_gen, scheduler_gen, loader,
             eval_loader, device, writer, trainer):
    losses = np.zeros(4, dtype=np.double)
    scheduler_gen.step()
    for i, ((x1, label1), (x2, label2)) in tqdm(enumerate(zip(loader, eval_loader))):
        x1 = x1.to(device)
        x2 = x2.to(device)
        z, x_ = model(x1)

        loss = loss_mod([x2], x_, z)
        # loss = [torch.nn.functional.mse_loss(x2, x_[0])]

        optimizer_gen.zero_grad()
        (loss[0]).backward(retain_graph=True)
        optimizer_gen.step()

        for j in range(len(loss)):
            losses[j] += loss[j].item() * 100

        writer.add_scalar('train/loss', loss[0], trainer.global_step)
        # writer.add_scalar('train/z0', z[0].item(), global_step)
        # writer.add_scalar('train/z1', z[1].item(), global_step)
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

            if plot_sample:
                plot_sample = False
                print(x1.mean(), x1.min(), x1.max(), x1.shape)
                writer.add_image('test/input',
                                 x1.add(1).mul(0.5).cpu(),
                                 trainer.global_step)
                writer.add_image('test/target',
                                 x2.add(1).mul(0.5).cpu(),
                                 trainer.global_step)
                writer.add_image('test/output',
                                 x_[0].add(1).mul(0.5).cpu(),
                                 trainer.global_step)

            all_z.append(z[0])
            all_angles.append(label2[:, 3])

            loss = loss_mod([x2], x_, z)
            # loss = [torch.nn.functional.mse_loss(x2, x_[0])]

            for i in range(len(loss)):
                losses[i] += loss[i].item() * 100

    all_z = torch.cat(all_z, dim=0)
    all_angles = torch.cat(all_angles, dim=0)

    fig, ax = plt.subplots()
    ax.scatter(all_angles, all_z[:, 0])
    ax.set(xlabel='$\\theta$', ylabel='z')
    writer.add_figure('test/z0', fig, trainer.global_step)

    fig, ax = plt.subplots()
    ax.scatter(all_angles, all_z[:, 1])
    ax.set(xlabel='$\\theta$', ylabel='z')
    writer.add_figure('test/z1', fig, trainer.global_step)

    return losses


if __name__ == "__main__":
    model = Model()
    loss_mod = Loss_Module(bootstrap_L2)
    optimizer = torch.optim.SGD(model.parameters(), 0.000001)
    sched = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.1)
    trainer = Trainer(None, None)
    trainer.train(model, 20, optimizer, sched, loss_mod, 'cuda')
