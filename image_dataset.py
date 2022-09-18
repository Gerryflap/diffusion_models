# This loader contains code from https://github.com/pytorch/vision/blob/master/torchvision/datasets/celeba.py
# It's designed to load an image dataset form the given folder
import PIL
from torchvision.datasets import VisionDataset
import os


class ImageDataset(VisionDataset):

    def __init__(self, folder, transform=None, target_transform=None):
        super().__init__(folder, transforms=None, transform=transform, target_transform=target_transform)

        self.folder = folder
        self.filename = [fname for fname in os.listdir(folder) if
                         fname.endswith(".png") or fname.endswith(".jpg") or fname.endswith(".jpeg")]

    def __getitem__(self, index):
        X = PIL.Image.open(os.path.join(self.folder, self.filename[index]))

        if self.transform is not None:
            X = self.transform(X)

        return X, []

    def __len__(self):
        return len(self.filename)