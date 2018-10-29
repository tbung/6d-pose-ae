from PIL import Image, ImageDraw
import numpy as np
import fire
import tqdm
from pathlib import Path


def render(n_samples, imsize, sqsize=None, fixed_z=True, allow_occlusion=False,
           root='./squares'):

    root = Path(root)
    root.mkdir(exist_ok=True)
    (root / 'no_rotation').mkdir(exist_ok=True)
    (root / 'no_translation').mkdir(exist_ok=True)
    (root / 'images').mkdir(exist_ok=True)

    scale = 1

    if not sqsize:
        sqsize = np.random.random_sample() * (0.5 - 0.1) * imsize + 0.1*imsize

    # Choose rectangle coordinates in allowed area
    if allow_occlusion:
        max_xy = imsize
        min_xy = 0
    else:
        max_xy = imsize - np.sqrt(2)*sqsize/2
        min_xy = np.sqrt(2)*sqsize/2

    gt = []

    for i in tqdm.trange(n_samples, dynamic_ncols=True):
        x0 = np.random.randint(min_xy, max_xy, 2)
        rect_xy = np.array([[0,           0],
                            [0,      sqsize],
                            [sqsize, sqsize],
                            [sqsize,      0],
                            ]) + x0 - sqsize/2

        img = Image.new('RGB', (imsize, imsize), 0)
        draw = ImageDraw.Draw(img)
        draw.polygon(list(rect_xy.flatten()), fill=(255, 255, 255))
        img.save(root / 'no_rotation' / f'{i:05d}.png')

        # Choose rotation in [0, pi) because symmetry
        theta = np.random.random_sample() * 0.5 * np.pi
        R = np.array([[np.cos(theta), np.sin(theta)],
                      [-np.sin(theta), np.cos(theta)]])

        rect_xy = (R @ (rect_xy - x0).T).T + imsize/2

        img = Image.new('RGB', (imsize, imsize), 0)
        draw = ImageDraw.Draw(img)
        draw.polygon(list(rect_xy.flatten()), fill=(255, 255, 255))
        img.save(root / 'no_translation' / f'{i:05d}.png')

        rect_xy += x0 - imsize/2

        # Draw image
        img = Image.new('RGB', (imsize, imsize), 0)
        draw = ImageDraw.Draw(img)
        draw.polygon(list(rect_xy.flatten()), fill=(255, 255, 255))
        img.save(root / 'images' / f'{i:05d}.png')

        gt.append([*x0, scale, theta])

    np.savetxt(root / 'target.txt', np.array(gt), delimiter='\t')


if __name__ == '__main__':
    fire.Fire(render)
