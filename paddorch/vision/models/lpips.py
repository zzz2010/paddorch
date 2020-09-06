"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import paddorch as torch
import paddorch.nn as nn
from paddorch.vision import models
# from torchvision import models
from paddle import fluid

def normalize(x, eps=1e-10):
    return x * torch.rsqrt(torch.sum(x**2, dim=1, keepdim=True) + eps)


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = models.alexnet(None).features
        self.channels = []
        for  name,layer in self.layers._sub_layers.items():
            if isinstance(layer, fluid.dygraph.Conv2D ):
                self.channels.append(layer._num_filters)

    def forward(self, x):
        fmaps = []
        for  name,layer in self.layers._sub_layers.items():
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                fmaps.append(x)
        return fmaps


class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False))

    def forward(self, x):
        return self.main(x)


class LPIPS(nn.Module):
    def __init__(self,pretrained_weights_fn=None):
        super().__init__()
        self.alexnet = AlexNet()
        self.lpips_weights = nn.ModuleList()
        for channels in self.alexnet.channels:
            self.lpips_weights.append(Conv1x1(channels, 1))
        if pretrained_weights_fn is not None:
            self._load_lpips_weights(pretrained_weights_fn)
        # imagenet normalization for range [-1, 1]
        self.mu = torch.tensor([-0.03, -0.088, -0.188]).view(1, 3, 1, 1)
        self.sigma = torch.tensor([0.458, 0.448, 0.450]).view(1, 3, 1, 1)

    def _load_lpips_weights(self,pretrained_weights_fn):
        own_state_dict = self.state_dict()
        state_dict = torch.load(pretrained_weights_fn)
        self.load_state_dict(state_dict)
        # for name, param in state_dict.items():
        #     if name in own_state_dict:
        #         own_state_dict[name]=torch.c


    def forward(self, x, y):
        x = (x - self.mu) / self.sigma
        y = (y - self.mu) / self.sigma
        x_fmaps = self.alexnet(x)
        y_fmaps = self.alexnet(y)

        lpips_value = 0
        for x_fmap, y_fmap, conv1x1 in zip(x_fmaps, y_fmaps, self.lpips_weights):
            x_fmap = normalize(x_fmap)
            y_fmap = normalize(y_fmap)
            z=torch.pow(x_fmap - y_fmap,2)
            lpips_value += torch.mean(conv1x1(z))
            # print("paddle alexnet mean", torch.mean(z).numpy(),lpips_value.numpy())
        return lpips_value


