import torch

from data_loader import get_loader

from utils import Codebook, symmetries, symmetries_diff, eulerAnglesToRotationMatrix
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import utils
import numpy as np
import pandas as pd
import trimesh

from sixd_toolkit.pysixd import pose_error


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


def vsd(all_pose_est, all_pose_gt, shape):
    K = np.array([[70, 0, 0], [0, 70, 0], [0, 0, 1]])
    if shape in ['cat', 'eggbox']:
        model = trimesh.load(f"./data/{shape}/mesh.ply")
        vert = model.vertices - model.center_mass
        vert /= np.abs(vert).max()
        vert *= 2
        faces = model.faces
    elif shape == "cube":
        p = np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1],
                      [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, -1]],
                     dtype=float)

        faces_p = [0, 1, 2, 3,  0, 3, 4, 5,   0, 5, 6, 1,
                   1, 6, 7, 2,  7, 4, 3, 2,   4, 7, 6, 5]

        vert = p[faces_p]

        faces = np.resize(
           np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32), 6 * (2 * 3))
        faces += np.repeat(4 * np.arange(6, dtype=np.uint32), 6)

    model = {"pts": vert, "faces": faces}
    errs = []
    i = 0
    for est, gt in tqdm(zip(all_pose_est, all_pose_gt)):
        if i == 200:
            break
        errs.append(pose_error.vsd(*est, *gt, model, 100*np.ones((128, 128)), K,
                                   10, 100, cost_type='step', shape=shape))
        i += 1

    errs = np.array(errs)

    print(f"VSD Mean: {errs.mean()}")
    print(f"VSD Recall: {len(errs[errs < 0.3])/len(errs)}")


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
    # print(codebook.rot.shape)

    loader_test = get_loader(f'./data/{args.dataset}', image_size=128,
                             batch_size=64, dataset='Geometric',
                             mode='test', num_workers=4,
                             pin_memory=True, mean=[0]*3,
                             std=[1]*3)

    length = len(loader_test)
    MSE_trans = torch.zeros(3).to(device)
    MSE_rot = torch.zeros(3).to(device)
    all_errors = []
    all_pose_est = []
    all_pose_gt = []
    cos_sim_mean = torch.zeros(1)
    cos_sim_std = torch.zeros(1)

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

        all_pose_est.extend(list(zip(rot_.cpu().numpy(), trans_.cpu().numpy())))
        all_pose_gt.extend(list(zip(rot.cpu().numpy(), trans.cpu().numpy())))

        with torch.no_grad():
            abs_diff = symmetries_diff(rot, rot_, object_type=args.dataset)
            # print(abs_diff.max())
            MSE_trans += ((trans-trans_)**2).mean(dim=0)/length
            MSE_rot += ((abs_diff)**2).mean(dim=0)/length

            cos_sim = utils.quaternion_distance(rot, rot_)
            cos_sim_mean += cos_sim.mean()/length
            cos_sim_std += cos_sim.std()/length

    print(f"RMSE trans: {torch.sqrt(MSE_trans).cpu().numpy()}")
    print(f"RMSE rot:   {torch.sqrt(MSE_rot).cpu().numpy()}")
    print(f"COS SIM mean: {cos_sim_mean.cpu().item()}")
    print(f"COS SIM std:   {cos_sim_std.cpu().item()}")

    vsd(all_pose_est, all_pose_gt, args.dataset)

# plt.scatter(codebook.z_rot[:, 0].cpu().numpy(), codebook.rot[:, 0].cpu().numpy())
# plt.show()
