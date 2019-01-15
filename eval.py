import torch

from data_loader import get_loader

from utils import Codebook, symmetries, symmetries_diff, eulerAnglesToRotationMatrix
import argparse
from tqdm import tqdm
import numpy as np
import trimesh

from sixd_toolkit.pysixd import pose_error

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


def vsd(all_pose_est, all_pose_gt):
    K = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    model = trimesh.load("./data/cat/mesh.ply")
    vert = model.vertices - model.center_mass
    vert /= np.abs(vert).max()
    vert *= 2
    model = {"pts": vert, "faces": model.faces}
    errs = []
    i = 0
    for est, gt in tqdm(zip(all_pose_est, all_pose_gt)):
        if i > 50:
            break
        errs.append(pose_error.vsd(*est, *gt, model, 100*np.ones((128, 128)), K,
                                   15, 30))
        i += 1

    errs = np.array(errs)

    print("VSD:")
    print(errs)
    print(errs.mean())
    print(errs.max())


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
                             batch_size=1, dataset='Geometric',
                             mode='test', num_workers=4,
                             pin_memory=True, mean=[0]*3,
                             std=[1]*3)

    length = len(loader_test)
    MSE_trans = torch.zeros(3).to(device)
    MSE_rot = torch.zeros(3).to(device)
    all_errors = []
    all_pose_est = []
    all_pose_gt = []

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

        all_pose_est.append((eulerAnglesToRotationMatrix(rot_),
                             trans_.cpu()))
        all_pose_gt.append((eulerAnglesToRotationMatrix(rot[0]),
                            trans.cpu()))
        with torch.no_grad():
            abs_diff = symmetries_diff(rot, rot_, object_type=args.dataset)
            print(abs_diff.max())
            MSE_trans += ((trans-trans_)**2).mean(dim=0)/length
            MSE_rot += ((abs_diff)**2).mean(dim=0)/length

    print(torch.sqrt(MSE_trans))
    print(torch.sqrt(MSE_rot))
