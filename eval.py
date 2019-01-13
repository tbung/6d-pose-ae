import torch

from data_loader import get_loader

from utils import Codebook, symmetries
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

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
    parser.add_argument('--model', type=str, default=None, required=True)
    args = parser.parse_args()

    model = torch.load(args.model)

    loader = get_loader(f'./data/{args.dataset}', image_size=128,
                        batch_size=64, dataset='Geometric',
                        mode='train', num_workers=4, pin_memory=True,
                        mean=[0]*3, std=[1]*3)

    codebook = Codebook(model, loader, device=device)
    print(codebook.rot.shape)

    loader_test = get_loader(f'./data/{args.dataset}', image_size=128,
                             batch_size=64, dataset='Geometric',
                             mode='test', num_workers=4,
                             pin_memory=True, mean=[0]*3,
                             std=[1]*3)

    length = len(loader_test)
    MSE_trans = torch.zeros(3).to(device)
    MSE_rot = torch.zeros(3).to(device)

    for i, (x, _1, _2, label) in tqdm(enumerate(loader_test)):
        x = x.to(device)
        label = label.to(device)
        trans = label[:, :3]//codebook.step_ax * codebook.step_ax
        rot = label[:, 3:]//codebook.step_rot * codebook.step_rot
        rot_, trans_ = codebook(x)
        rot_ = rot_.squeeze()
        # print(rot_.shape)
        # print(rot.shape)
        # print(trans_.shape)
        # if i == 0:
        # print ('rot',rot[:20])
        # print('trans',trans[:20])
        rot = symmetries(rot, object_type=args.dataset)
        rot_ = symmetries(rot_, object_type=args.dataset)
        print(torch.abs(rot-rot_).max())
        with torch.no_grad():
            MSE_trans += ((trans-trans_)**2).mean(dim=0)/length
            MSE_rot += ((rot-rot_)**2).mean(dim=0)/length

    print(MSE_trans)
    print(MSE_rot)

plt.scatter(codebook.z_rot[:, 0].cpu().numpy(), codebook.rot[:, 0].cpu().numpy())
plt.show()
