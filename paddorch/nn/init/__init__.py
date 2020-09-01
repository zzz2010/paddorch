import paddle.fluid as fluid
import paddle
import paddle_torch.cuda
import paddle_torch.nn
import os
import paddle_torch.nn.functional
from paddle.fluid import dygraph
import numpy as np


def constant_(x, val):
    x=fluid.layers.fill_constant(x.shape,x.dtype,val,out=x)
    return x

