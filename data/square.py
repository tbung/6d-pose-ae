from PIL import Image, ImageDraw
import numpy as np
import fire
import tqdm
from pathlib import Path


def render(n_samples, imsize, sqsize=None, allow_occlusion=False,
           root='./squares'):

    root = Path(root)
    root.mkdir(exist_ok=True)

    if not sqsize:
        sqsize = np.random.random_sample() * (0.5 - 0.1) * imsize + 0.1*imsize

    # Choose rectangle coordinates in allowed area
    if allow_occlusion:
        max_xy = imsize
        min_xy = 0
    else:
        max_xy = imsize - np.sqrt(2)*sqsize/2
        min_xy = np.sqrt(2)*sqsize/2

    for i in tqdm.trange(n_samples, dynamic_ncols=True):
        x0 = np.random.randint(min_xy, max_xy, 2)
        rect_xy = np.array([[0,           0],
                            [0,      sqsize],
                            [sqsize, sqsize],
                            [sqsize,      0],
                            ]) + x0 - sqsize/2

        # Choose rotation in [0, pi) because symmetry
        theta = np.random.random_sample() * 0.5 * np.pi
        R = np.array([[np.cos(theta), np.sin(theta)],
                      [-np.sin(theta), np.cos(theta)]])

        rect_xy = (R @ (rect_xy - x0).T).T + x0

        # Draw image
        img = Image.new('RGB', (imsize, imsize), 0)
        draw = ImageDraw.Draw(img)
        draw.polygon(list(rect_xy.flatten()), fill=(255, 255, 255))
        img.save(root / f'{i:05d}.png')


if __name__ == '__main__':
    fire.Fire(render)
