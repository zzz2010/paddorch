import nn
import nn.functional as F
import numpy as np
import math
import paddle.fluid as fluid
import paddorch as torch
from paddorch.vision.models.lpips import LPIPS



def eval_pytorch_model():
    import   torch
    import torch.nn as nn
    from torchvision import models
    # from torchvision import models

    def normalize(x, eps=1e-10):
        return x * torch.rsqrt(torch.sum(x ** 2, dim=1, keepdim=True) + eps)

    class AlexNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = models.alexnet(pretrained=True).features
            self.channels = []

            for layer in self.layers:
                if isinstance(layer, nn.Conv2d):
                    self.channels.append(layer.out_channels)

        def forward(self, x):
            fmaps = []
            for layer in self.layers:
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
        def __init__(self):
            super().__init__()
            self.alexnet = AlexNet()
            self.lpips_weights = nn.ModuleList()
            for channels in self.alexnet.channels:
                self.lpips_weights.append(Conv1x1(channels, 1))
            self._load_lpips_weights()
            # imagenet normalization for range [-1, 1]
            self.mu = torch.tensor([-0.03, -0.088, -0.188]).view(1, 3, 1, 1).
            self.sigma = torch.tensor([0.458, 0.448, 0.450]).view(1, 3, 1, 1).

        def _load_lpips_weights(self):
            own_state_dict = self.state_dict()

            state_dict = torch.load('lpips_weights.ckpt',
                                        map_location=torch.device('cpu'))
            for name, param in state_dict.items():
                if name in own_state_dict:
                    own_state_dict[name].copy_(param)

        def forward(self, x, y):
            x = (x - self.mu) / self.sigma
            y = (y - self.mu) / self.sigma
            x_fmaps = self.alexnet(x)
            y_fmaps = self.alexnet(y)

            lpips_value = 0
            for x_fmap, y_fmap, conv1x1 in zip(x_fmaps, y_fmaps, self.lpips_weights):
                x_fmap = normalize(x_fmap)
                y_fmap = normalize(y_fmap)
                z = torch.pow(x_fmap - y_fmap, 2)
                lpips_value += torch.mean(conv1x1(z))
                # lpips_value += torch.mean(conv1x1((x_fmap - y_fmap) ** 2))
                # print("torch alexnet mean", torch.mean(z),lpips_value)
            return lpips_value

    return LPIPS()



if __name__ == '__main__':
    from paddorch.convert_pretrain_model import load_pytorch_pretrain_model
    import torch as pytorch
    import torchvision
    pytorch_model=eval_pytorch_model()
    pytorch_model.
    place = fluid.CPUPlace()
    x=np.ones((1,3,256,256)).astype("float32")
    y = np.zeros((1, 3, 256, 256)).astype("float32")
    pytorch_model.eval()
    torch_output=pytorch_model(pytorch.FloatTensor(x).,pytorch.FloatTensor(y).).detach().numpy()

    pytorch_model

    with fluid.dygraph.guard(place=place):
        model=LPIPS()
        pytorch_state_dict=pytorch_model.state_dict()
        load_pytorch_pretrain_model(model, pytorch_state_dict)
        torch.save(model.state_dict(),"LPIPS_pretrained")
        model.eval()
        paddle_output=model(torch.Tensor(x),torch.Tensor(y)).numpy()
        print("paddle output  ",paddle_output)
        print("torch output  ", torch_output)