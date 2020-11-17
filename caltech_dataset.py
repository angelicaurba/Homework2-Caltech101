from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def read_file(filename, root):
    file = open(filename, 'r')
    Lines = file.readlines()

    count = 0
    lab = 0
    labels = {}
    images = []
    for line in Lines:
        path = line.strip()
        parts = line.split("/")
        if parts[0] == 'BACKGROUND_Google':
            continue
        label = labels.get(parts[0], -1)
        if label == -1:
            labels[parts[0]] = lab
            label = lab
            lab+=1
        image = pil_loader(root+"/"+path)
        images.append([image, label])
        count+=1

    return images, labels, count


class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')
        files_root = "/".join(root.split("/")[:-1])
        files_root+="/"

        if split == 'train':
            filename = files_root+"train.txt"
        elif split == 'test':
            filename = files_root+"test.txt"
        else: raise Exception("no allowed split passed")

        self.images, self.labels, self.length = read_file(filename, root)

    def __getitem__(self, index):

        image, label = (self.images[index][0], self.images[index][1])

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        length = self.length
        return length
