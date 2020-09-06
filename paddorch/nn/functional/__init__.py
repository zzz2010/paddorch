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

def dropout(input, p=0.5, training=True, inplace=False):
    return torch.Tensor(fluid.layers.dropout(input,
            p,
            is_test=not training,
         dropout_implementation='upscale_in_train'))

def softmax(input, dim=None, _stacklevel=3, dtype=None):
    return torch.Tensor(fluid.layers.softmax(input,axis=dim))

def embedding(x, weight):
    return fluid.layers.embedding(fluid.layers.reshape(x,[-1,1]), size=weight.shape,param_attr=NumpyArrayInitializer(weight.numpy()))


def batch_norm(x,  running_mean, running_var, weight=None, bias=None,
               training=False, momentum=0.1, eps=1e-5):
    layer_object=fluid.dygraph.BatchNorm(x.shape[1],momentum=momentum,epsilon=eps,trainable_statistics=training)
    fluid.layers.assign(running_mean,layer_object._mean)
    fluid.layers.assign(running_var, layer_object._variance)
    if weight is not None:
        fluid.layers.assign(weight, layer_object.weight)
    if bias is not None:
        fluid.layers.assign(bias, layer_object.bias)
    return layer_object(x)



#TODO: need to do unit test to confirm this function
def linear(input, weight, bias=None):
    layer_obj=fluid.dygraph.Linear(input.shape[1],weight.shape[1])
    fluid.layers.assign(weight,layer_obj.weight)
    if bias is not None:
        fluid.layers.assign(bias, layer_obj.bias)
    return layer_obj(input)

def normalize(input, p=2, dim=1, eps=1e-12, out=None):
    return torch.Tensor(fluid.layers.l2_normalize(input,axis=dim,epsilon=eps))
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
    else:
        bias_attr=NumpyArrayInitializer(bias.numpy())
    return fluid.layers.conv2d(x , num_filters=weight.shape[0],filter_size=weight.shape[-2:],stride=stride,padding=padding,dilation=dilation,groups=groups,
            param_attr=fluid.ParamAttr(initializer=NumpyArrayInitializer(weight.numpy())),bias_attr=bias_attr)


# from torch.nn.functional import  l1_loss,mse_loss,binary_cross_entropy_with_logits
#
# def l1_loss(input, target, size_average=None, reduce=None, reduction='mean'):
#     return fluid.dygraph.L1Loss()