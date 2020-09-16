
import numpy as np
import math
import paddle.fluid as fluid
import paddorch as torch

from paddorch.vision.models.wing import FAN, preprocess



def eval_pytorch_model():
    import   torch
    import torch.nn as nn
    import torch.nn.functional   as F
    from torchvision import models
    from functools import partial
    # from torchvision import models

    def get_preds_fromhm(hm):
        max, idx = torch.max(
            hm.view(hm.size(0), hm.size(1), hm.size(2) * hm.size(3)), 2)
        idx += 1
        preds = idx.view(idx.size(0), idx.size(1), 1).repeat(1, 1, 2).float()
        preds[..., 0].apply_(lambda x: (x - 1) % hm.size(3) + 1)
        preds[..., 1].add_(-1).div_(hm.size(2)).floor_().add_(1)

        for i in range(preds.size(0)):
            for j in range(preds.size(1)):
                hm_ = hm[i, j, :]
                pX, pY = int(preds[i, j, 0]) - 1, int(preds[i, j, 1]) - 1
                if pX > 0 and pX < 63 and pY > 0 and pY < 63:
                    diff = torch.FloatTensor(
                        [hm_[pY, pX + 1] - hm_[pY, pX - 1],
                         hm_[pY + 1, pX] - hm_[pY - 1, pX]])
                    preds[i, j].add_(diff.sign_().mul_(.25))

        preds.add_(-0.5)
        return preds

    class HourGlass(nn.Module):
        def __init__(self, num_modules, depth, num_features, first_one=False):
            super(HourGlass, self).__init__()
            self.num_modules = num_modules
            self.depth = depth
            self.features = num_features
            self.coordconv = CoordConvTh(64, 64, True, True, 256, first_one,
                                         out_channels=256,
                                         kernel_size=1, stride=1, padding=0)
            self._generate_network(self.depth)

        def _generate_network(self, level):
            self.add_module('b1_' + str(level), ConvBlock(256, 256))
            self.add_module('b2_' + str(level), ConvBlock(256, 256))
            if level > 1:
                self._generate_network(level - 1)
            else:
                self.add_module('b2_plus_' + str(level), ConvBlock(256, 256))
            self.add_module('b3_' + str(level), ConvBlock(256, 256))

        def _forward(self, level, inp):
            up1 = inp
            up1 = self._modules['b1_' + str(level)](up1)
            low1 = F.avg_pool2d(inp, 2, stride=2)
            low1 = self._modules['b2_' + str(level)](low1)

            if level > 1:
                low2 = self._forward(level - 1, low1)
            else:
                low2 = low1
                low2 = self._modules['b2_plus_' + str(level)](low2)
            low3 = low2
            low3 = self._modules['b3_' + str(level)](low3)
            up2 = F.interpolate(low3, scale_factor=2, mode='nearest')

            return up1 + up2

        def forward(self, x, heatmap):
            x, last_channel = self.coordconv(x, heatmap)
            return self._forward(self.depth, x), last_channel

    class AddCoordsTh(nn.Module):
        def __init__(self, height=64, width=64, with_r=False, with_boundary=False):
            super(AddCoordsTh, self).__init__()
            self.with_r = with_r
            self.with_boundary = with_boundary
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            with torch.no_grad():
                x_coords = torch.arange(height).unsqueeze(1).expand(height, width).float()
                y_coords = torch.arange(width).unsqueeze(0).expand(height, width).float()
                x_coords = (x_coords / (height - 1)) * 2 - 1
                y_coords = (y_coords / (width - 1)) * 2 - 1
                coords = torch.stack([x_coords, y_coords], dim=0)  # (2, height, width)

                if self.with_r:
                    rr = torch.sqrt(torch.pow(x_coords, 2) + torch.pow(y_coords, 2))  # (height, width)
                    rr = (rr / torch.max(rr)).unsqueeze(0)
                    coords = torch.cat([coords, rr], dim=0)

                self.coords = coords.unsqueeze(0).to(device)  # (1, 2 or 3, height, width)
                self.x_coords = x_coords.to(device)
                self.y_coords = y_coords.to(device)

        def forward(self, x, heatmap=None):
            """
            x: (batch, c, x_dim, y_dim)
            """
            coords = self.coords.repeat(x.size(0), 1, 1, 1)

            if self.with_boundary and heatmap is not None:
                boundary_channel = torch.clamp(heatmap[:, -1:, :, :], 0.0, 1.0)
                zero_tensor = torch.zeros_like(self.x_coords)
                xx_boundary_channel = torch.where(boundary_channel > 0.05, self.x_coords, zero_tensor).to(
                    zero_tensor.device)
                yy_boundary_channel = torch.where(boundary_channel > 0.05, self.y_coords, zero_tensor).to(
                    zero_tensor.device)
                coords = torch.cat([coords, xx_boundary_channel, yy_boundary_channel], dim=1)

            x_and_coords = torch.cat([x, coords], dim=1)
            return x_and_coords

    class CoordConvTh(nn.Module):
        """CoordConv layer as in the paper."""

        def __init__(self, height, width, with_r, with_boundary,
                     in_channels, first_one=False, *args, **kwargs):
            super(CoordConvTh, self).__init__()
            self.addcoords = AddCoordsTh(height, width, with_r, with_boundary)
            in_channels += 2
            if with_r:
                in_channels += 1
            if with_boundary and not first_one:
                in_channels += 2
            self.conv = nn.Conv2d(in_channels=in_channels, *args, **kwargs)

        def forward(self, input_tensor, heatmap=None):
            ret = self.addcoords(input_tensor, heatmap)
            last_channel = ret[:, -2:, :, :]
            ret = self.conv(ret)
            return ret, last_channel

    class ConvBlock(nn.Module):
        def __init__(self, in_planes, out_planes):
            super(ConvBlock, self).__init__()
            self.bn1 = nn.BatchNorm2d(in_planes)
            conv3x3 = partial(nn.Conv2d, kernel_size=3, stride=1, padding=1, bias=False, dilation=1)
            self.conv1 = conv3x3(in_planes, int(out_planes / 2))
            self.bn2 = nn.BatchNorm2d(int(out_planes / 2))
            self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
            self.bn3 = nn.BatchNorm2d(int(out_planes / 4))
            self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))

            self.downsample = None
            if in_planes != out_planes:
                self.downsample = nn.Sequential(nn.BatchNorm2d(in_planes),
                                                nn.ReLU(True),
                                                nn.Conv2d(in_planes, out_planes, 1, 1, bias=False))

        def forward(self, x):
            residual = x

            out1 = self.bn1(x)
            out1 = F.relu(out1, True)
            out1 = self.conv1(out1)

            out2 = self.bn2(out1)
            out2 = F.relu(out2, True)
            out2 = self.conv2(out2)

            out3 = self.bn3(out2)
            out3 = F.relu(out3, True)
            out3 = self.conv3(out3)

            out3 = torch.cat((out1, out2, out3), 1)
            if self.downsample is not None:
                residual = self.downsample(residual)
            out3 += residual
            return out3

    class FAN(nn.Module):
        def __init__(self, num_modules=1, end_relu=False, num_landmarks=98, fname_pretrained=None):
            super(FAN, self).__init__()
            self.num_modules = num_modules
            self.end_relu = end_relu

            # Base part
            self.conv1 = CoordConvTh(256, 256, True, False,
                                     in_channels=3, out_channels=64,
                                     kernel_size=7, stride=2, padding=3)
            self.bn1 = nn.BatchNorm2d(64)
            self.conv2 = ConvBlock(64, 128)
            self.conv3 = ConvBlock(128, 128)
            self.conv4 = ConvBlock(128, 256)

            # Stacking part
            self.add_module('m0', HourGlass(1, 4, 256, first_one=True))
            self.add_module('top_m_0', ConvBlock(256, 256))
            self.add_module('conv_last0', nn.Conv2d(256, 256, 1, 1, 0))
            self.add_module('bn_end0', nn.BatchNorm2d(256))
            self.add_module('l0', nn.Conv2d(256, num_landmarks + 1, 1, 1, 0))

            if fname_pretrained is not None:
                self.load_pretrained_weights(fname_pretrained)

        def load_pretrained_weights(self, fname):
            if torch.cuda.is_available():
                checkpoint = torch.load(fname)
            else:
                checkpoint = torch.load(fname, map_location=torch.device('cpu'))
            model_weights = self.state_dict()
            model_weights.update({k: v for k, v in checkpoint['state_dict'].items()
                                  if k in model_weights})
            self.load_state_dict(model_weights)

        def forward(self, x):
            x, _ = self.conv1(x)
            x = F.relu(self.bn1(x), True)
            x = F.avg_pool2d(self.conv2(x), 2, stride=2)
            x = self.conv3(x)
            x = self.conv4(x)

            outputs = []
            boundary_channels = []
            tmp_out = None
            ll, boundary_channel = self._modules['m0'](x, tmp_out)
            ll = self._modules['top_m_0'](ll)
            ll = F.relu(self._modules['bn_end0']
                        (self._modules['conv_last0'](ll)), True)

            # Predict heatmaps
            tmp_out = self._modules['l0'](ll)
            if self.end_relu:
                tmp_out = F.relu(tmp_out)  # HACK: Added relu
            outputs.append(tmp_out)
            boundary_channels.append(boundary_channel)
            return outputs, boundary_channels

        def get_heatmap(self, x, b_preprocess=True):
            x.stop_gradient = True
            ''' outputs 0-1 normalized heatmap '''
            x = F.interpolate(x, size=256, mode='bilinear')
            x_01 = x * 0.5 + 0.5
            outputs, _ = self(x_01)
            heatmaps = outputs[-1][:, :-1, :, :]
            scale_factor = x.size(2) // heatmaps.size(2)
            if b_preprocess:
                heatmaps = F.interpolate(heatmaps, scale_factor=scale_factor,
                                         mode='bilinear', align_corners=True)
                heatmaps = preprocess(heatmaps)
            return heatmaps

        def get_landmark(self, x):
            x.stop_gradient = True
            ''' outputs landmarks of x.shape '''
            heatmaps = self.get_heatmap(x, b_preprocess=False)
            landmarks = []
            for i in range(x.size(0)):
                pred_landmarks = get_preds_fromhm(heatmaps[i].unsqueeze(0))
                landmarks.append(pred_landmarks)
            scale_factor = x.size(2) // heatmaps.size(2)
            landmarks = torch.cat(landmarks) * scale_factor
            return landmarks




    return FAN(fname_pretrained="../../../starganv2_paddle/expr/checkpoints/wing.pt")



if __name__ == '__main__':
    from paddorch.convert_pretrain_model import load_pytorch_pretrain_model
    import torch as pytorch
    import torchvision

    # place = fluid.CPUPlace()
    place = fluid.CUDAPlace(0)
    np.random.seed(0)
    x=np.random.randn(11,3,256,256).astype("float32")
    with fluid.dygraph.guard(place=place):
        model=FAN()
        model.eval()
        pytorch_model=eval_pytorch_model()
        pytorch_model.eval()
        pytorch_model.cuda()
        torch_output = pytorch_model(pytorch.FloatTensor(x).cuda())
        pytorch_state_dict=pytorch_model.state_dict()
        load_pytorch_pretrain_model(model, pytorch_state_dict)
        torch.save(model.state_dict(),"wing")
        paddle_output = model(torch.Tensor(x))

        print("torch mean",torch_output[0][0].shape,torch_output[0][0].mean().cpu().detach().numpy())
        print("paddle mean",paddle_output[0][0].shape, torch.mean(paddle_output[0][0]).numpy())

        print("torch mean",torch_output[1][0].shape,torch_output[1][0].mean().cpu().detach().numpy())
        print("paddle mean",paddle_output[1][0].shape, torch.mean(paddle_output[1][0]).numpy())