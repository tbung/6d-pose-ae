from torch.utils import data
from torchvision import transforms as T
from PIL import Image
import torch
import os
import random

###FIX AMOUNT OF DATA LOADED DURCH BATCH###
class GeometricDataset(data.Dataset):
    """Dataset class for gemoetric shapes."""

    def __init__(self, image_dir, attr_path, transform, selected_attrs = None, mode = 'train'):
        """Init and preprocess dataset"""
        self.image_dir  = image_dir
        self.attr_path  = attr_path
        self.selected_attrs  = selected_attrs
        self.transform  = transform
        self.mode       = mode
        self.train_dataset  = []
        self.test_dataset   = []
        self.attr2idx   = {}
        self.idx2attr   = {}
        self.preprocess()
        

        if mode == 'train':
            self.num_images     = len(self.train_dataset)

        else:
            self.num_images     = len(self.test_dataset)

    def preprocess(self):
        """Prepocess the geometric attribute file """
        lines   = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names  = lines[0].split()

        size    = 0
        for i, attr_name in enumerate(all_attr_names):
            if i == 0:
                size    = int(attr_name)
                print(size)
                continue
            k = i -1
            self.attr2idx[attr_name] = k
            self.idx2attr[k] = attr_name

        if self.selected_attrs is None: self.selected_attrs = all_attr_names[1:]

        lines = lines[1:]
        random.seed(1)
        random.shuffle(lines)
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(float(values[idx]))

            if (i) < size // 4:
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])
    print('Finished preprocessing the Geometric dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images

def get_loader(image_dir, attr_path, selected_attrs = None, image_size=64, 
               batch_size=16, dataset='Geometric', mode='train', num_workers=4, pin_memory = True, 
               mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]):
    """Build and return a data loader."""
    transform = []
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean = mean, std = std))
    transform = T.Compose(transform)

    if dataset == 'Geometric':
        dataset = GeometricDataset(image_dir, attr_path, transform, selected_attrs, mode)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle= False,
                                  num_workers= num_workers,
                                  pin_memory = pin_memory)
    return data_loader


def main():
    test_laod = get_loader('./data/square/images', './data/square/target.txt', image_size= 64, batch_size= 12, mode= 'train')
    test_img, test_label = next(iter(test_laod))
    print("Shape of the image batch")
    print(test_img.shape)
    print("Shape of the label batch")
    print(test_label.shape)
    print("First Label")
    print(test_label[0])
    print("Amount of images in train")
    print(len(test_laod.dataset))

if __name__ == "__main__":
    main()

