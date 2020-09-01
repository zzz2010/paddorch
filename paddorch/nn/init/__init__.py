import paddle.fluid as fluid
import paddle
import paddorch.cuda
import paddorch.nn
import os
import paddorch.nn.functional
from paddle.fluid import dygraph
import numpy as np


def constant_(x, val):
    x=fluid.layers.fill_constant(x.shape,x.dtype,val,out=x)
    return x

