import torch
import copy
import os
import torchvision.utils as vutils
from data_loader import get_loader
from utils import *
from tqdm import tqdm





class Trainer:
    def __init__(self, image_dir, attr_path):
        self.loader     = get_loader(image_dir, attr_path, selected_attrs = None, image_size=64, 
               batch_size=64, dataset='Geometric', mode='train', num_workers=4, pin_memory = True)
        self.rot_loader = get_loader(image_dir, attr_path, selected_attrs = None, image_size=64, 
               batch_size=64, dataset='Geometric', mode='train', num_workers=4, pin_memory = True)
        self.loader_test= get_loader(image_dir, attr_path, selected_attrs = None, image_size=64, 
               batch_size=64, dataset='Geometric', mode='test', num_workers=4, pin_memory = True)
        self.rot_loader_test = get_loader(image_dir, attr_path, selected_attrs = None, image_size=64, 
               batch_size=64, dataset='Geometric', mode='test', num_workers=4, pin_memory = True)
        #self.dist_loader= get_loader(image_dir, attr_path, selected_attrs = None, image_size=64, 
        #       batch_size=64, dataset='Geometric', mode='train', num_workers=4, pin_memory = True)
        self.mean       = 0.5
        self.std        = 0.5
        self.f_epoch    = 1 / len(self.loader)
        self.f_eval     = 1 / len(self.loader_test)
        

    def train(self, model, epochs, optimizer, scheduler,loss_mod, device, save_dir = None):
        model   = model.to(device)
        loss_mod    = loss_mod.to(device)


        if save_dir is not None:
            loss_array = []
            path_im = os.path.join('./images', 'ae', save_dir)
            path_run= os.path.join('./saved_models', 'ae', save_dir)
            create_folder(path_im)
            create_folder(path_run)

        for epoch in range(1, epochs+1):
            model.train()
            print('_______________')
            losses = ae_epoch(model, loss_mod, optimizer, scheduler, 
                        self.loader, self.rot_loader, device)
            
            losses = self.f_epoch * losses
            print(
                    'Epoch: {} \n'
                    'Training: \n'
                    'Loss: {:.3f} \t L_rec: {:.3f} \t'
                    'L_sparse:{:.3f} \t L_distr: {:.3f}'.format(
                        epoch ,losses[0], losses[1], losses[2], losses[3]))

            model.eval()
            
            losses_test = ae_eval(model, tracker, loss_mod, 
                        self.loader_test, self.rot_loader_test, device)
            
            losses_test = self.f_eval * losses_test
            print(
                    '\n'
                    'Test: \n'
                    'Loss: {:.3f} \t L_rec: {:.3f} \t'
                    'L_sparse:{:.3f} \t L_distr: {:.3f}'.format(
                        losses_test[0], losses_test[1], losses_test[2], losses_test[3]))

            if save_dir is not None:
                with torch.no_grad():
                    loss_array.append(np.append(losses, losses_test))


                    x           = next(iter(self.loader))[0].to(device)
                    _, x_   = model(x) 
                    x_          = x_ * self.std + self.mean
                    x_rec       =   x_[:,:3].clamp(0,1).to('cpu')
                    vutils.save_image(x_rec, os.path.join(path_im,'{}_epoch_rec.png'.format(epoch))) 
                    torch.save(model.state_dict(), os.path.join(path_run, '{}_model.pth'.format(epoch))) 

        if save_dir is not None:
            loss_array = np.stack(loss_array, axis = 0)
            save_log(path_im, loss_array)
        return model

        
def ae_epoch(model, loss_mod, optimizer_gen, scheduler_gen, 
                    loader, eval_loader, device):
    losses = np.zeros(4, dtype = np.double)
    scheduler_gen.step()
    for (x1, label1),(x2, label2) in tqdm(zip(loader, eval_loader)):

            
            x1      = x1.to(device)
            x2      = x2.to(device)
            z, x_   = model(x1) 

            
            l           = loss_mod(x2, x_, z)



            optimizer_gen.zero_grad()
            (l[0]).backward(retain_graph = True)
            optimizer_gen.step()


            for i in range(len(l)):
                losses[i] += l[i].item() * 100

    return losses


def ae_eval(model, tracker, loss_mod, loader, eval_loader, device):
    losses = np.zeros(4, dtype = np.double)
    tracker.reset()
    for (x1, label1),(x2, label2) in tqdm(zip(loader, eval_loader)):
        with torch.no_grad():
            x1      = x1.to(device)
            x2      = x2.to(device)
            z, x_   = model(x1) 

            
            l           = loss_mod(x2, x_, z)

            for i in range(len(l)):
                losses[i] += l[i].item() * 100

            tracker.update(z)
    return losses