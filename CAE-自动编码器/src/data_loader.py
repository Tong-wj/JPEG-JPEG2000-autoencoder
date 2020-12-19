import glob
import torchvision.transforms as transforms

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageFolder96p(Dataset):
    """
    Image shape is (96, 96, 3)  --> 1x1 128x128 x 3 patches
    """

    def __init__(self, folder_path):
        self.files = sorted(glob.glob('%s/*.*' % folder_path))

    def __getitem__(self, index):
        path = self.files[index % len(self.files)]
        img = Image.open(path)
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            # transforms.Normalize([0.4431991, 0.42826223, 0.39535823], [0.25746644, 0.25306803, 0.26591763])
        ])
        img = transform(img)
        img = np.array(img)
        # img = np.array(Image.open(path)) / 255.0
        h, w, c = img.shape

        # 使用了ToTensor,就不需要下面一行的转换了
        # img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float()

        patches = np.reshape(img, (3, 1, 128, 1, 128))
        patches = np.transpose(patches, (0, 1, 3, 2, 4))

        return img, patches, path

    def get_random(self):
        i = np.random.randint(0, len(self.files))
        return self[i]

    def __len__(self):
        return len(self.files)
