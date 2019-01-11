import torch
from model import Model

from data_loader import get_loader

from utils import Codebook
import argparse
from tqdm import tqdm

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        choices=['square', 'cube', 'cat', 'eggbox'],
                        default='square', required=False)
    parser.add_argument('--mode', type=str,
                        choices=['both', 'no_trans', 'no_rot'],
                        default='both', required=False)
    parser.add_argument('--trans-dim', type=int, default=3, required=False)
    parser.add_argument('--rot-dim', type=int, default=4, required=False)
    parser.add_argument('--state_dict', type=str, default = None, required=False)
    args = parser.parse_args()

    model = Model(trans_dim=args.trans_dim, rot_dim=args.rot_dim, w=128)

    if args.state_dict:
        model.load_state_dict(torch.load(args.state_dict))

    loader = get_loader(f'./data/{args.dataset}', image_size=128,
                                 batch_size=64, dataset='Geometric',
                                 mode='train', num_workers=4, pin_memory=True,
                                 mean=[0]*3, std=[1]*3)


    codebook = Codebook(model, loader, device  =device )
    #print(codebook.rot.shape)

    loader_test = get_loader(f'./data/{args.dataset}', image_size=128,
                                      batch_size=64, dataset='Geometric',
                                      mode='test', num_workers=4,
                                      pin_memory=True, mean=[0]*3,
                                      std=[1]*3)


    length = len(loader_test)
    MSE_trans   = torch.zeros(3)
    MSE_rot     = torch.zeros(2)

    for i, (x, _1, _2, label) in tqdm(enumerate(loader_test)):
        trans = label[:, :3]//codebook.step_ax * codebook.step_ax
        rot = label[:,3:]//codebook.step_rot * codebook.step_rot
        rot_, trans_ = codebook(x)
        rot_ = rot_.squeeze()
        print(rot_.shape)
        print(rot.shape)
        print(trans_.shape)
        #if i == 0:
            #print ('rot',rot[:20])
            #print('trans',trans[:20])
        with torch.no_grad():
            MSE_trans += ((trans-trans_)**2).mean(dim=0)/length
            MSE_rot += ((rot-rot_)**2).mean(dim=0)/length

    print(MSE_trans)
    print(MSE_rot)
         
        
