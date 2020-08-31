import nn
import nn.functional as F
import numpy as np
import math
import paddle.fluid as fluid
import paddle_torch as torch
from paddle_torch.vision.models.inception import InceptionV3



def eval_pytorch_model():
    import   torch
    import torch.nn as nn
    from torchvision import models
    # from torchvision import models

    class InceptionV3(nn.Module):
        def __init__(self):
            super().__init__()
            inception = models.inception_v3(pretrained=True)
            self.block1 = nn.Sequential(
                inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3,
                inception.Conv2d_2b_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2))
            self.block2 = nn.Sequential(
                inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2))
            self.block3 = nn.Sequential(
                inception.Mixed_5b, inception.Mixed_5c,
                inception.Mixed_5d, inception.Mixed_6a,
                inception.Mixed_6b, inception.Mixed_6c,
                inception.Mixed_6d, inception.Mixed_6e)
            self.block4 = nn.Sequential(
                inception.Mixed_7a, inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1)))

        def forward(self, x):
            x = self.block1(x)
            print("torch block1",x.mean())
            x = self.block2(x)
            print("torch block2", x.mean())
            x = self.block3(x)
            print("torch block3", x.mean())
            x = self.block4(x)
            print("torch block4", x.mean())
            return x.view(x.size(0), -1)

    return InceptionV3()



if __name__ == '__main__':

    from paddle_torch.convert_pretrain_model import load_pytorch_pretrain_model
    import torch as pytorch
    import torchvision
    import sys




    # place = fluid.CPUPlace()
    place = fluid.CUDAPlace(0)
    with fluid.dygraph.guard(place=place):
        x = np.random.randn(1, 3, 256, 256)

        print("paddle:", torch.nn.functional.avg_pool2d(torch.Tensor(x),   kernel_size=3, stride=1, padding=1).numpy().mean())
        print("torch:", pytorch.nn.functional.avg_pool2d(pytorch.FloatTensor(x), kernel_size=3, stride=1, padding=1).numpy().mean())

        # sys.exit()

        model=InceptionV3()
        model.eval()
        pytorch_model=eval_pytorch_model()
        pytorch_model.eval()
        pytorch_model.cuda()
        x=np.ones((1,3,256,256)).astype("float32")
        torch_output=pytorch_model(pytorch.FloatTensor(x).cuda())
        pytorch_model.cpu()
        pytorch_state_dict=pytorch_model.state_dict()
        load_pytorch_pretrain_model(model, pytorch_state_dict)
        torch.save(model.state_dict(),"inception_v3_pretrained")


        paddle_output=model(torch.Tensor(x))

        print("torch mean",torch_output.mean())
        print("paddle mean", torch.mean(paddle_output).numpy())