import paddle.fluid as fluid
from paddle.fluid.initializer import NumpyArrayInitializer

import paddorch
import paddorch as torch
from paddle.nn.functional import softplus,nll_loss
import paddle

def avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
    if stride is None:
        stride=kernel_size

    return torch.Tensor(fluid.layers.pool2d(input,
                           pool_size=kernel_size, pool_type="avg", pool_stride=stride,
                                            pool_padding=padding, global_pooling=False, use_cudnn=True,
                                            ceil_mode=ceil_mode, name=None, exclusive=not count_include_pad, data_format="NCHW"))

def max_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
    if stride is None:
        stride=kernel_size

    return torch.Tensor(fluid.layers.pool2d(input,
                           pool_size=kernel_size, pool_type="max", pool_stride=stride, pool_padding=padding,
                                            global_pooling=False, use_cudnn=True, ceil_mode=ceil_mode, name=None,
                                            exclusive=not count_include_pad, data_format="NCHW"))

def tanh(x):
    return torch.Tensor(fluid.layers.tanh(x))

def dropout(input, p=0.5, training=True, inplace=False):
    return torch.Tensor(fluid.layers.dropout(input,
            p,
            is_test=not training,
         dropout_implementation='upscale_in_train'))

def softmax(input, dim=None, _stacklevel=3, dtype=None):
    return torch.Tensor(fluid.layers.softmax(input,axis=dim))

def embedding(x, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False):


    layer=fluid.dygraph.Embedding( size=weight.shape,is_sparse=sparse,padding_idx=padding_idx)

    fluid.layers.assign(weight,layer.weight)
    out = layer(x)
    # if norm_type is not None:
    #     out=paddle.nn.functional.normalize(out, p=norm_type, axis=-1)

    return out



def batch_norm(x,  running_mean, running_var, weight=None, bias=None,
               training=False, momentum=0.1, eps=1e-5):
    layer_object=fluid.dygraph.BatchNorm(x.shape[1],momentum=momentum,epsilon=eps,trainable_statistics=training)
    fluid.layers.assign(running_mean,layer_object._mean)
    fluid.layers.assign(running_var, layer_object._variance)
    if weight is not None:
        fluid.layers.assign(weight, layer_object.weight)
    if bias is not None:
        fluid.layers.assign(bias, layer_object.bias)
    return torch.Tensor(layer_object(x))



#TODO: need to do unit test to confirm this function
def linear(input, weight, bias=None):
    if input.shape[-1]!=weight.shape[0]:
        weight=paddle.transpose(weight,[1,0])

    layer_obj=fluid.dygraph.Linear(input.shape[-1],weight.shape[1])
    fluid.layers.assign(weight,layer_obj.weight)
    if bias is not None:
        fluid.layers.assign(bias, layer_obj.bias)
    return paddorch.convertTensor(layer_obj(input.astype("float32")))

def normalize(input, p=2, dim=1, eps=1e-12, out=None):
    return torch.convertTensor( input/paddle.norm(input,p,axis=dim,keepdim=True))
    # return torch.Tensor(fluid.layers.l2_normalize(input,axis=dim,epsilon=eps))
def sigmoid(x):
    return torch.Tensor(fluid.layers.sigmoid(x))

def binary_cross_entropy_with_logits(logits, targets):
    return fluid.layers.sigmoid_cross_entropy_with_logits(logits, targets)

def adaptive_avg_pool2d(input, output_size):
    return torch.Tensor(fluid.layers.adaptive_pool2d(input,pool_size=output_size,pool_type="avg"))
def adaptive_max_pool2d(input, output_size):
    return torch.Tensor(fluid.layers.adaptive_pool2d(input,pool_size=output_size,pool_type="max"))

def leaky_relu(input, negative_slope=0.01, inplace=False):
    return  torch.Tensor(fluid.layers.leaky_relu(input, alpha=negative_slope, name=None))

def relu(input,inplace=False):
    return torch.Tensor(fluid.layers.relu(input))

def interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=False,align_mode=1,data_format='NCHW'):
    if isinstance(size,int):
        size=[size,size]
    return torch.Tensor(fluid.layers.interpolate(input,
                out_shape=size,
                scale=scale_factor,
                name=None,
                resample=mode.upper(),
                actual_shape=None,
                align_corners=align_corners,
                align_mode=align_mode,
                data_format=data_format))

def conv2d(input, weight, bias=None, stride=1, padding=1,dilation=1, groups=1):
    if bias is None:
        bias_attr=False
    else:
        bias_attr=None

    layer=fluid.dygraph.Conv2D(num_channels=weight.shape[1], num_filters=weight.shape[0],filter_size=weight.shape[-2:],stride=stride,padding=padding,dilation=dilation,groups=groups,bias_attr=bias_attr)
    # layer.weight.set_value(weight)
    fluid.layers.assign(weight,layer.weight)
    if bias is not None:
        # layer.bias.set_value(bias)
        fluid.layers.assign(bias, layer.bias)
    out=layer(input)
    return out



def conv_transpose2d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    if bias is None:
        bias_attr=False
    else:
        bias_attr=None

    layer=fluid.dygraph.Conv2DTranspose(num_channels=weight.shape[0], num_filters=weight.shape[1],filter_size=weight.shape[-2:],stride=stride,padding=padding,dilation=dilation,groups=groups,bias_attr=bias_attr)
    # layer.weight.set_value(weight)
    fluid.layers.assign(weight,layer.weight)
    if bias is not None:
        # layer.bias.set_value(bias)
        fluid.layers.assign(bias, layer.bias)
    out=layer(input)
    return out


# from torch.nn.functional import  l1_loss,mse_loss,binary_cross_entropy_with_logits
#
# def l1_loss(input, target, size_average=None, reduce=None, reduction='mean'):
#     return fluid.dygraph.L1Loss()



def elu(x):
    return fluid.layers.elu(x, alpha=1.0, name=None)


def cross_entropy(input, label):
    return paddle.nn.functional.cross_entropy(input, label, weight=None, ignore_index=- 100, reduction='mean',
                                              soft_label=False, axis=- 1, name=None)


def cosine_similarity(x1, x2):
    return paddle.nn.functional.cosine_similarity(x1, x2, axis=1, eps=1e-8)


def log_softmax(x, dim):
    return paddle.nn.functional.log_softmax(x, axis=dim, dtype=None, name=None)


def pad(input, pad, mode='constant', value=0):
    pad2=[]
    for _ in range(len(input.shape)*2-len(pad)):
        pad2.append(0)
    if isinstance(pad, tuple):
        pad=list(pad)
    pad2=pad2+pad
    return paddle.nn.functional.pad(input,pad2,mode=mode,value=value)