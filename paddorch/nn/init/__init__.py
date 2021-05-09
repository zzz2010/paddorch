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

def _calculate_correct_fan(x, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(x)
    return fan_in if mode == 'fan_in' else fan_out

def kaiming_normal_(x,a=0,  mode='fan_in',nonlinearity='leaky_relu'):
    fan = _calculate_correct_fan(x, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    with paddle.no_grad():
        return normal_(x,0, std)

#TODO find the right implementation
def xavier_normal_(tensor, gain=1.0):
    return kaiming_normal_(tensor)*gain

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
    if len(x.shape)==1:
        fan_in, fan_out =x.shape[0],1
    else:
        fan_in, fan_out = _calculate_fan_in_and_fan_out(x)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

    return uniform_(x, -a, a)

def uniform_(x, a=0, b=1.):
    temp_value = paddle.uniform(min=a, max=b, shape=x.shape)
    x.set_value(temp_value)
    return x



def zeros_(x):
    temp_value = paddle.zeros(x.shape, dtype=None, name=None)
    x.set_value(temp_value)
    return x

def ones_(x):
    temp_value = paddle.ones(x.shape, dtype=None, name=None)
    x.set_value(temp_value)
    return x

def calculate_gain(nonlinearity, param=None):
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    elif nonlinearity == 'selu':
        return 3.0 / 4  # Value found empirically (https://github.com/pytorch/pytorch/pull/50664)
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))