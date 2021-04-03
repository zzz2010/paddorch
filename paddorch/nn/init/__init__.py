import paddle.fluid as fluid
import paddle
import paddorch.cuda
import paddorch.nn
import paddorch.nn.functional
import math


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


def _calculate_fan_in_and_fan_out(tensor):
    dimensions = len(tensor.shape)
    if dimensions < 2:
        raise ValueError(
            "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
        )

    num_input_fmaps = tensor.shape[1]
    num_output_fmaps = tensor.shape[0]
    receptive_field_size = 1
    if len(tensor.shape) > 2:
        receptive_field_size = paddle.numel(tensor[0][0])
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out

def xavier_uniform_(x, gain=1.):
    """Fills the input `Tensor` with values according to the method
    described in `Understanding the difficulty of training deep feedforward
    neural networks` - Glorot, X. & Bengio, Y. (2010), using a uniform
    distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-a, a)` where
    .. math::
        a = \text{gain} \times \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}
    Also known as Glorot initialization.
    Args:
        x: an n-dimensional `paddle.Tensor`
        gain: an optional scaling factor
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(x)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

    return uniform_(x, -a, a)

def uniform_(x, a=-1., b=1.):
    temp_value = paddle.uniform(min=a, max=b, shape=x.shape)
    x.set_value(temp_value)
    return x