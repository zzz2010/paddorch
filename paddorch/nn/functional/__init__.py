import paddle.fluid as fluid
from paddle.fluid.initializer import NumpyArrayInitializer
import paddorch as torch

def avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
    if stride is None:
        stride=kernel_size

    return fluid.layers.pool2d(input,
                           pool_size=kernel_size, pool_type="avg", pool_stride=stride, pool_padding=padding, global_pooling=False, use_cudnn=True, ceil_mode=ceil_mode, name=None, exclusive=not count_include_pad, data_format="NCHW")

def max_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
    if stride is None:
        stride=kernel_size

    return fluid.layers.pool2d(input,
                           pool_size=kernel_size, pool_type="max", pool_stride=stride, pool_padding=padding, global_pooling=False, use_cudnn=True, ceil_mode=ceil_mode, name=None, exclusive=not count_include_pad, data_format="NCHW")

def tanh(x):
    return fluid.layers.tanh(x)

def sigmoid(x):
    return fluid.layers.sigmoid(x)

def binary_cross_entropy_with_logits(logits, targets):
    return fluid.layers.sigmoid_cross_entropy_with_logits(logits, targets)

def adaptive_avg_pool2d(input, output_size):
    return torch.Tensor(fluid.layers.adaptive_pool2d(input,pool_size=output_size,pool_type="avg"))
def adaptive_max_pool2d(input, output_size):
    return torch.Tensor(fluid.layers.adaptive_pool2d(input,pool_size=output_size,pool_type="max"))

def leaky_relu(input, negative_slope=0.01, inplace=False):
    return  fluid.layers.leaky_relu(input, alpha=negative_slope, name=None)

def relu(input,inplace=False):
    return fluid.layers.relu(input)

def interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=False,align_mode=1,data_format='NCHW'):
    if isinstance(size,int):
        size=[size,size]
    return fluid.layers.interpolate(input,
                out_shape=size,
                scale=scale_factor,
                name=None,
                resample=mode.upper(),
                actual_shape=None,
                align_corners=align_corners,
                align_mode=align_mode,
                data_format=data_format)

def conv2d(x, weight, bias=None, stride=1, padding=1,dilation=1, groups=1):
    if bias is None:
        bias_attr=False
    return fluid.layers.conv2d(x , num_filters=weight.shape[0],filter_size=weight.shape[-2:],stride=stride,padding=padding,dilation=dilation,groups=groups,
            param_attr=fluid.ParamAttr(initializer=NumpyArrayInitializer(weight.numpy())),bias_attr=bias_attr)


# from torch.nn.functional import  l1_loss,mse_loss,binary_cross_entropy_with_logits
#
# def l1_loss(input, target, size_average=None, reduce=None, reduction='mean'):
#     return fluid.dygraph.L1Loss()