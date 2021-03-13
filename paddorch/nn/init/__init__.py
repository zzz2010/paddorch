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

def normal_(x,m=0,std=1):
    y=paddle.randn(x.shape)*std+m
    fluid.layers.assign(y, x)
    # fluid.layers.assign(np.random.randn(*x.shape).astype(np.float32)*std+m,x)
    return x

def kaiming_normal_(x,nonlinearity=None):
    ##didnt know how to implement correctly, use normal as  placeholder
    x= normal_(x)
    if nonlinearity is not None:
        if nonlinearity =="relu":
            x=paddorch.nn.functional.relu(x)
        if nonlinearity =="tanh":
            x=paddorch.nn.functional.tanh(x)
    return x

def constant_(x,val):
    y=fluid.layers.zeros(x.shape,"float32")+val
    fluid.layers.assign(y, x)
    return x
