"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
import argparse

import paddorch as torch
import paddorch.nn as nn
import numpy as np
from paddorch.vision.models.inception import InceptionV3
from scipy import linalg

from PIL import Image
import numpy as np

from pathlib import Path
from itertools import chain
import os
from paddle import fluid
import random
import paddorch as torch
from paddorch.utils import data
from paddorch.utils.data.sampler import WeightedRandomSampler
from paddorch.vision import transforms
from paddorch.vision.datasets import ImageFolder




def listdir(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in ['png', 'jpg', 'jpeg', 'JPG']]))
    return fnames


class DefaultDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.samples = listdir(root)
        self.samples.sort()
        self.transform = transform
        self.targets = None

    def __getitem__(self, index):
        fname = self.samples[index]
        img = np.array(Image.open(fname).convert('RGB'))
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)


try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x): return x



def frechet_distance(mu, cov, mu2, cov2):
    cc, _ = linalg.sqrtm(np.dot(cov, cov2), disp=False)
    dist = np.sum((mu -mu2)**2) + np.trace(cov + cov2 - 2*cc)
    return np.real(dist)


def get_eval_loader(root, img_size=256, batch_size=32,
                    imagenet_normalize=True, shuffle=True,
                    num_workers=4, drop_last=False):
    print('Preparing DataLoader for the evaluation phase...')
    if imagenet_normalize:
        height, width = 299, 299
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        height, width = img_size, img_size
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.Resize([height, width]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    dataset = DefaultDataset(root, transform=transform)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=drop_last)


@torch.no_grad()
def calculate_fid_given_paths(paths, img_size=256, batch_size=50,inception_pretrain_fn="../../metrics/inception_v3_pretrained.pdparams"):
    print('Calculating FID given paths %s and %s...' % (paths[0], paths[1]))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inception =   InceptionV3(inception_pretrain_fn)
    inception.eval()
    loaders = [get_eval_loader(path, img_size, batch_size) for path in paths]

    mu, cov = [], []
    for loader in loaders:
        actvs = []
        for x in tqdm(loader, total=len(loader)):
            x=torch.varbase_to_tensor(x[0])
            actv = inception(x )
            actvs.append(actv)
        actvs = torch.cat(actvs, dim=0).numpy()
        mu.append(np.mean(actvs, axis=0))
        cov.append(np.cov(actvs, rowvar=False))
    fid_value = frechet_distance(mu[0], cov[0], mu[1], cov[1])
    return fid_value.astype(float)


if __name__ == '__main__':
    """
    --paths ../../data/afhq/val ../../expr/eval/afhq
    --pretrain ../../metrics/inception_v3_pretrained.pdparams
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths', type=str, nargs=2, help='paths to real and fake images')
    parser.add_argument('--img_size', type=int, default=256, help='image resolution')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size to use')
    parser.add_argument('--pretrain', type=str,  help='path InceptionV3 pretrain model')
    args = parser.parse_args()
    place = fluid.CUDAPlace(0)
    with fluid.dygraph.guard(place=place):
        fid_value = calculate_fid_given_paths(args.paths, args.img_size, args.batch_size,args.pretrain)
    print('FID: ', fid_value)

# python -m metrics.fid --paths PATH_REAL PATH_FAKE