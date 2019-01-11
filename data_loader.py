from torch.utils import data
from torchvision import transforms as T
from PIL import Image
import torch
from pathlib import Path
import pandas as pd


class GeometricDataset(data.Dataset):
    """Dataset class for gemoetric shapes."""

    def __init__(self, root, transform, mode='train'):
        """Init and preprocess dataset"""
        self.root = Path(root)
        self.image_dirs = [
            self.root / "images",
            self.root / "no_translation",
            self.root / "no_rotation",
        ]
        self.transform = transform
        self.mode = mode

        self.dataset = pd.read_table(self.root / "target.txt")

        # No shuffle before split to prevent items being in both train and test
        # set
        if mode == "train":
            self.dataset = self.dataset.iloc[len(self.dataset)//4:]
        elif mode == "test":
            self.dataset = self.dataset.iloc[:len(self.dataset)//4]

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        filename, *labels = self.dataset.iloc[index]
        images = [Image.open(p / filename) for p in self.image_dirs]
        images = list(map(self.transform, images))
        return (*images, torch.FloatTensor(labels))

    def __len__(self):
        """Return the number of images."""
        return len(self.dataset)


def get_loader(image_dir, image_size=64, batch_size=16, dataset='Geometric',
               mode='train', num_workers=4, pin_memory=True,
               mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    """Build and return a data loader."""
    transform = []
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=mean, std=std))
    transform = T.Compose(transform)

    if dataset == 'Geometric':
        dataset = GeometricDataset(image_dir, transform, mode)

    shuffle = True if mode == "train" else False

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


if __name__ == "__main__":
    test_laod = get_loader('./data/square', image_size=64, batch_size=12,
                           mode='train')
    test_img, t_t, t_r, test_label = next(iter(test_laod))
    print("Shape of the image batch")
    print(test_img.shape)
    print("Shape of the label batch")
    print(test_label.shape)
    print("First Label")
    print(test_label[0])
    print("Amount of images in train")
    print(len(test_laod.dataset))
