import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from pathlib import Path
from tqdm import tqdm

from data_loader import get_loader


def printfig(filename=None):
    if filename is not None:
        plt.savefig('{}.pgf'.format(filename))
        plt.savefig('{}.pdf'.format(filename), format="PDF")
    else:
        plt.show()


def latexify(fig_width=None, fig_height=None):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    """

    fig_width_pt = 483.69684   # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27  # Convert pt to inch

    if fig_width is None:
        fig_width = fig_width_pt*inches_per_pt

    if fig_height is None:
        golden_mean = (np.sqrt(5)-1.0)/2.0    # Aesthetic ratio
        fig_height = fig_width*golden_mean # height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:" + fig_height +
              "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
        fig_height = MAX_HEIGHT_INCHES

    params = {
      'backend': 'ps',
      #'text.latex.preamble': ['\usepackage{gensymb}'],
      'axes.labelsize': 8, # fontsize for x and y labels (was 10)
      'axes.titlesize': 8,
      'font.size':       8, # was 10
      'legend.fontsize': 8, # was 10
      'xtick.labelsize': 8,
      'ytick.labelsize': 8,
      'text.usetex': True,
      'figure.figsize': [fig_width,fig_height],
      'font.family': 'serif'
    }

    matplotlib.rcParams.update(params)



def interpolate():
    model = torch.load('./runs/Jan11_14-32-04_GLaDOS_square_both/checkpoints/70.model')
    angles = torch.tensor(np.linspace(np.pi/2, 1.5*np.pi, 64)).float()
    z = torch.stack([torch.cos(angles),  torch.sin(angles)], dim=1)
    save_image(model.dec1(z), './figures/interpolation_theta.png')

    x = torch.tensor(np.linspace(0, 2, 64)).float()
    y = torch.zeros(64)

    lat = torch.stack([x, y, y], dim=1).float()
    save_image(model.dec2(lat), './figures/interpolation_x.png', pad_value=0.5)

    lat = torch.stack([y, x, y], dim=1).float()
    save_image(model.dec2(lat), './figures/interpolation_y.png', pad_value=0.5)

    lat = torch.stack([y, y, x], dim=1).float()
    save_image(model.dec2(lat), './figures/interpolation_z.png', pad_value=0.5)


def correlation_plots():
    device = 'cuda'
    for shape in ['square', 'cube', 'cat', 'eggbox']:
        model = torch.load(sorted(Path('./runs').glob(f'*{shape}*'))[-1] /
                           'checkpoints/70.model')
        model.to(device)
        loader = get_loader(f'./data/{shape}', image_size=128,
                            batch_size=64, dataset='Geometric',
                            mode='test', num_workers=4,
                            pin_memory=True, mean=[0]*3,
                            std=[1]*3)
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
                z, x_ = model(x1, mode='both')

                val_x = [x2, x3]
                out_x = x_

                if plot_sample:
                    plot_sample = False
                    save_image(x1.cpu(), f'./figures/{shape}_input.png', pad_value=0.5)
                    for i, x in enumerate(out_x):
                        save_image(x.cpu(), f'./figures/{shape}_output{i}.png', pad_value=0.5)
                        save_image(val_x[i].cpu(), f'./figures/{shape}_target{i}.png', pad_value=0.5)

                all_z.append(z[0])
                all_angles.append(label[:, 3:])
                all_axis.append(label[:, :3])
                all_z_ax.append(z[1])

        all_z = torch.cat(all_z, dim=0).cpu()
        all_z_ax = torch.cat(all_z_ax, dim=0).cpu()
        all_angles = torch.cat(all_angles, dim=0).cpu()
        all_axis = torch.cat(all_axis, dim=0).cpu()

        for i in range(model.trans_dim):
            for j, name in enumerate(['x', 'y', 'z']):
                fig, ax = plt.subplots()
                ax.scatter(all_axis[:, j], all_z_ax[:, i], s=2)
                ax.set(xlabel=f'${name}$', ylabel=f'$z_{i}$')
                printfig(f'./figures/{shape}_{name}_z{i}')
                plt.close()
        for i in range(model.rot_dim):
            for j, name in enumerate(['theta', 'phi', 'gamma']):
                fig, ax = plt.subplots()
                ax.scatter(all_angles[:, j], all_z[:, i], s=2)
                ax.set(xlabel=f'$\\{name}$', ylabel=f'$z_{i}$')
                printfig(f'./figures/{shape}_{name}_z{i}')
                plt.close()


if __name__ == '__main__':
    latexify()
    interpolate()
    correlation_plots()
