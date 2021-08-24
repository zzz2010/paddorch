import paddle.fluid as fluid
from paddle.fluid import dygraph
from paddle.fluid.dygraph import layers, Conv2D, Linear, InstanceNorm
from . import functional as F
from .parameter import Parameter
import numpy as np
import paddorch.nn.utils
from . import init
from ..tensor import Tensor, convertTensor
import collections
import math
import sys
from functools import partial, reduce
import numbers
import paddle
import paddle.fluid as fluid
from paddle import framework
import itertools

from paddle.nn import initializer as I
from paddle.fluid.dygraph import Layer, LayerList
# from paddle.fluid.layers import utils
from paddle.fluid.layers.utils import map_structure, flatten, pack_sequence_as
import six

from paddle.fluid.dygraph import layers
from paddle.framework import get_default_dtype, set_default_dtype
from paddle.fluid.framework import in_dygraph_mode

from paddle.fluid.initializer import Constant
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.data_feeder import check_variable_and_dtype, check_type
from paddle.fluid import core, dygraph_utils

from paddle.nn.functional import batch_norm, layer_norm, instance_norm
from paddle.nn import Sequential
from collections import OrderedDict as ModuleDict
import warnings
from paddle.nn import functional as F
from paddle.nn import LogSigmoid,CrossEntropyLoss,ParameterList,PReLU,NLLLoss,KLDivLoss,GELU


def clone_layer(layer):
    new_obj=Layer()
    for name, layer in layer._sub_layers.items():
        new_obj.add_sublayer(name, clone_layer(layer))
    return new_obj


def forward_post_hook(layer,input,output):
    if isinstance(output,tuple):
        return tuple([convertTensor(x) if isinstance(x,dygraph.core.VarBase) else x for x in output])
    else:
        if isinstance(output,dygraph.core.VarBase) and not isinstance(output,Tensor):
            return convertTensor(output)
        else:
            return output

def forward_pre_hook(layer,input):
    if isinstance(input,tuple):
        return tuple([convertTensor(x) if isinstance(x,dygraph.core.VarBase) else x for x in input])
    else:
        if isinstance(input,dygraph.core.VarBase) and not isinstance(input,Tensor):
            return convertTensor(input)
        else:
            return input

class Module(Layer):
    def __init__(self , name_scope=None, dtype=core.VarDesc.VarType.FP32):
        super(Module, self).__init__(name_scope,dtype)
        # self.register_buffer=dict()

        self.register_forward_post_hook(forward_post_hook)
        self.register_forward_pre_hook(forward_pre_hook)


    def eval(self):
        super(Module, self).eval()
        return self
    def load_state_dict(self,new_dict, strict=True):
        self.set_dict(new_dict,   use_structured_name=True)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        self.set_dict(state_dict, use_structured_name=True)


    def named_modules(self,
                        memo=None, prefix='', remove_duplicate=True):

        return  super().named_sublayers(prefix=prefix)

    def register_parameter(self,name,value):
        self.__setattr__(name,Parameter(value))

    def register_buffer(self,name,value):
        '''state dict will record this value, but no training on it'''
        if value is None   : ##do the deletion
            self.__setattr__(name, None)
            # if hasattr(self,name):
            #     delattr(self,name)
            return
        X=Parameter(value)
        X.stop_gradient=True
        self.__setattr__(name,X)
    def add_module(self,name,layer):
        return self.add_sublayer(name,layer)

    def modules(self):
        return self.sublayers()

    # def __getattr__(self, name):
    #     try:
    #         return super(Module, self).__getattr__()
    #     except:
    #         return None

    def clone(self):
        import copy
        new_obj= Module()
        for name,layer in self._sub_layers.items():
            new_obj.add_sublayer(name,clone_layer(layer) )
        for name,params in self._parameters.items() :
            new_obj.add_parameter(name,copy.deepcopy(params) )
        new_obj.load_state_dict(self.state_dict())
        return new_obj
    def to(self,device=None):
        return self

    def cuda(self):
        return self

    def cpu(self):
        return self


    def reset_parameters(self):
        for pp in self.parameters():
            paddorch.nn.init.xavier_uniform_(pp )


def DataParallel(model):
    return fluid.dygraph.DataParallel(model)
class Conv2d(dygraph.Conv2D,Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        bias_attr = None
        if not bias:
            bias_attr = False
        else:
            bias_attr =fluid.initializer.MSRAInitializer() # fluid.initializer.ConstantInitializer(value=0)

        super(Conv2d, self).__init__(num_channels=in_channels,
                                     num_filters=out_channels,
                                     filter_size=kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     dilation=dilation,
                                     groups=groups,
                                     param_attr=fluid.initializer.MSRAInitializer(),
                                     bias_attr=bias_attr,
                                     use_cudnn=True,
                                     act=None,
                                     dtype='float32')



class Conv1d(paddle.nn.Conv1D,Module):
    def __init__(self,*args,**kwargs):
        super(Conv1d, self).__init__(*args,**kwargs)

class InstanceNorm2d(Module):
    '''
    this version allow bias_attrs=False
    '''
    def __init__(self,
                 num_features, eps=1e-5, momentum=0.1, affine=False,
                 track_running_stats=False):
        super(InstanceNorm2d, self).__init__()

        self._epsilon = eps
        self._param_attr = None
        self._bias_attr = None
        self._dtype = 'float32'

        if   affine:
            self.scale = self.create_parameter(
                attr=self._param_attr,
                shape=[num_features],
                dtype=self._dtype,
                default_initializer=fluid.initializer.Constant(1.0),
                is_bias=False)
            self.bias = self.create_parameter(
                attr=self._bias_attr,
                shape=[num_features],
                dtype=self._dtype,
                default_initializer=fluid.initializer.Constant(0.0),
                is_bias=True)
        else:
            self.scale= fluid.layers.ones([num_features],dtype='float32')
            self.bias=  fluid.layers.zeros([num_features],dtype='float32')

    def forward(self, input):
        if in_dygraph_mode():
            out, _, _ = core.ops.instance_norm(input, self.scale, self.bias,
                                               'epsilon', self._epsilon)
            return out

        check_variable_and_dtype(input, 'input', ['float32', 'float64'],
                                 "InstanceNorm")

        attrs = {"epsilon": self._epsilon}

        inputs = {"X": [input], "Scale": [self.scale], "Bias": [self.bias]}

        saved_mean = self._helper.create_variable_for_type_inference(
            dtype=self._dtype, stop_gradient=True)
        saved_variance = self._helper.create_variable_for_type_inference(
            dtype=self._dtype, stop_gradient=True)
        instance_norm_out = self._helper.create_variable_for_type_inference(
            self._dtype)

        outputs = {
            "Y": [instance_norm_out],
            "SavedMean": [saved_mean],
            "SavedVariance": [saved_variance]
        }

        self._helper.append_op(
            type="instance_norm", inputs=inputs, outputs=outputs, attrs=attrs)
        return instance_norm_out
#
class Linear(dygraph.Linear,Module):
    def __init__(self,in_features, out_features, bias=True):
        uniform_bound=np.sqrt(1.0/in_features)
        param_attr=fluid.initializer.UniformInitializer(-uniform_bound,uniform_bound )
        if not bias:
            bias_attr = False
        else:
            bias_attr =fluid.initializer.UniformInitializer(-uniform_bound,uniform_bound )
        super(Linear, self).__init__(in_features, out_features, param_attr=param_attr, bias_attr=bias_attr, act=None, dtype="float32")



class LSTM(paddle.nn.LSTM,Module):
    '''    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
            would mean stacking two LSTMs together to form a `stacked LSTM`,
            with the second LSTM taking in outputs of the first LSTM and
            computing the final results. Default: 1
        bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
            Default: ``True``
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            LSTM layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0
        bidirectional: If ``True``, becomes a bidirectional LSTM. Default: ``False``'''

    def __init__(self,input_size,hidden_size,num_layers=1,bias=True,batch_first=False,dropout=0,bidirectional=False):
        bias_attr = None
        if not bias:
            bias_attr = False
        else:
            bias_attr =fluid.initializer.MSRAInitializer()
        if not bidirectional:
            direction="forward"
        else:
            direction="bidirectional"
        super(LSTM,self).__init__ ( input_size,
                 hidden_size,
                 num_layers=num_layers,
                 direction=direction,
                 time_major=not batch_first,
                 dropout=dropout,
                 weight_ih_attr=None,
                 weight_hh_attr=None,
                 bias_ih_attr=bias_attr,
                 bias_hh_attr=bias_attr,
                 name=None)

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)
class GRU(paddle.nn.GRU,Module):

    '''    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
            would mean stacking two GRUs together to form a `stacked GRU`,
            with the second GRU taking in outputs of the first GRU and
            computing the final results. Default: 1
        bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
            Default: ``True``
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            GRU layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0
        bidirectional: If ``True``, becomes a bidirectional GRU. Default: ``False``'''
    def __init__(self,input_size,hidden_size,num_layers=1,bias=True,batch_first=False,dropout=0,bidirectional=False):
        bias_attr = None
        if not bias:
            bias_attr = False
        else:
            bias_attr =fluid.initializer.MSRAInitializer()
        if not bidirectional:
            direction="forward"
        else:
            direction="bidirectional"
        super(GRU,self).__init__(  input_size,
                 hidden_size,
                 num_layers=num_layers,
                 direction=direction,
                 time_major=not batch_first,
                 dropout=dropout,
                 weight_ih_attr=None,
                 weight_hh_attr=None,
                 bias_ih_attr=bias_attr,
                 bias_hh_attr=bias_attr,
                 name=None)





class Embedding(paddle.nn.Embedding,Module):
    def __init__(self,num_embeddings: int, embedding_dim: int,
                 padding_idx  = None, max_norm = None, norm_type: float = 2.0, scale_grad_by_freq: bool = False,
                 sparse: bool = False, _weight = None):
        super(Embedding,self).__init__( num_embeddings,
                 embedding_dim,
                 padding_idx=padding_idx,
                 sparse=sparse,
                 weight_attr=_weight,
                 name=None
          )
        self.norm_type=norm_type
        self.max_norm=max_norm
        self.padding_idx=padding_idx


    def forward(self, input):
        if self.max_norm is not None:
            max_norm=paddorch.norm(self.weight, p=self.norm_type, keepdim=True)
            max_norm=paddorch.clamp(max_norm,0,self.max_norm)
            normalized_weight=self.weight / max_norm
            paddorch.copy(normalized_weight,self.weight )
        y=super(Embedding, self).forward(input)
        return y

# def Dropout(p=0.5, inplace=False):
#     paddle.nn.Dropout
#     return dygraph.Dropout(p,dropout_implementation='upscale_in_train')

class Dropout(paddle.nn.Dropout,Module):
    def __init__(self,p=0.5, inplace=False,):
        super().__init__(p=p )

class Upsample(Module):
    """
    This op resizes a batch of images.
    The input must be a 3-D Tensor of the shape (num_batches, channels, in_w)
    or 4-D (num_batches, channels, in_h, in_w), or a 5-D Tensor of the shape
    (num_batches, channels, in_d, in_h, in_w) or (num_batches, in_d, in_h, in_w, channels),
    and the resizing only applies on the three dimensions(depth, height and width).

            import paddle.fluid.dygraph as dg
            upsample_op = paddle.nn.UpSample(size=[12,12])
            input_data = np.random.rand(2,3,6,10).astype("float32")
            place = paddle.fluid.CPUPlace()
            with dg.guard(place) as g:
                input = dg.to_variable(input_data)
                output = upsample_op(input=input)
                print(output.shape)
                # [2L, 3L, 12L, 12L]
    """

    def __init__(self,
                 size=None,
                 scale_factor=None,
                 mode='nearest',
                 align_corners=False,
                 align_mode=1,
                 data_format='NCHW'):
        super(Upsample, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode.lower()
        self.align_corners = align_corners
        self.align_mode = align_mode
        self.data_format = data_format

    def forward(self, input):
        out = F.interpolate(
            input,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
            align_mode=self.align_mode,
            data_format=self.data_format)

        return out

class ConvTranspose2d(fluid.dygraph.Conv2DTranspose):

    def __init__(self, in_channels: int,
        out_channels: int,
        kernel_size,
        stride = 1,
        padding = 0,
        output_padding = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int = 1,
        padding_mode: str = 'zeros'):
        if bias == False:
            bias_attr=False
        else:
            bias_attr=None
        super(ConvTranspose2d,self).__init__(in_channels,
                 out_channels,
                 kernel_size,
                 output_size=None,
                 padding=padding,
                 stride=stride,
                 dilation=dilation,
                 groups=groups,
                 param_attr=None,
                 bias_attr=bias_attr,
                 use_cudnn=True,
                 act=None,
                 dtype='float32')



class ReLU(Module):
    def __init__(self, inplace=False):
        super(ReLU, self).__init__()


    def forward(self, input):
        return F.relu(input )

class AvgPool2d(Module):
    def __init__(self, kernel_size,inplace=False):
        super(AvgPool2d, self).__init__()
        self.kernel_size=kernel_size


    def forward(self, input):
        return F.avg_pool2d(input,self.kernel_size )

class Tanh(Module):
    def __init__(self, inplace=False):
        super(Tanh, self).__init__()


    def forward(self, input):
        return F.tanh(input )


class Softmax(Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim=dim
    def forward(self, input):
        return F.softmax(input,axis=self.dim )
class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        super(MaxPool2d, self).__init__()
        self.padding=padding
        self.kernel_size=kernel_size
        self.stride=stride
        self.dilation=dilation
        self.ceil_mode=ceil_mode
    def forward(self, input):
        return F.max_pool2d(input , kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, ceil_mode=self.ceil_mode, count_include_pad=True)


class AdaptiveAvgPool2d(Module):
    def __init__(self,output_size):
        super(AdaptiveAvgPool2d, self).__init__()
        self.output_size=output_size

    def forward(self, input):
        return F.adaptive_avg_pool2d(input, self.output_size)



class BatchNorm2d(dygraph.BatchNorm):
    def __init__(self,num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        param_attr = None
        bias_attr = None
        if not affine:
            param_attr = False
            bias_attr = False


        super(BatchNorm2d, self).__init__(num_features,
                 act=None,
                 is_test=False,
                 momentum=momentum,
                 epsilon=eps,
                 param_attr=param_attr,
                 bias_attr=bias_attr,
                 dtype='float32',
                 data_layout='NCHW',
                 in_place=False,
                 moving_mean_name=None,
                 moving_variance_name=None,
                 do_model_average_for_mean_and_var=True,
                 use_global_stats=False,
                 trainable_statistics=False)

#
# def BatchNorm2d(num_features, eps=1e-5, momentum=0.1, affine=True,
#                  track_running_stats=True):
#     param_attr=None
#     bias_attr=None
#     if not affine:
#         param_attr=False
#         bias_attr=False
#     return dygraph.BatchNorm(num_features,
#                  act=None,
#                  is_test=False,
#                  momentum=momentum,
#                  epsilon=eps,
#                  param_attr=param_attr,
#                  bias_attr=bias_attr,
#                  dtype='float32',
#                  data_layout='NCHW',
#                  in_place=False,
#                  moving_mean_name=None,
#                  moving_variance_name=None,
#                  do_model_average_for_mean_and_var=True,
#                  use_global_stats=False,
#                  trainable_statistics=False)

class LeakyReLU(Module):
    def __init__(self,negative_slope=1e-2, inplace=False):
        super(LeakyReLU, self).__init__()
        self.negative_slope=negative_slope

    def forward(self, input):

        return F.leaky_relu(input, negative_slope=self.negative_slope )


class ModuleList(dygraph.LayerList):
    def __init__(self, sublayers=None):
        super(ModuleList, self).__init__(sublayers)

    def __add__(self, other):
        for a in other:
            self.append(a)
        return self

    def insert(self, index, sublayer):
        """
        Insert a sublayer before a given index in the list. support case len(self._sub_layers)==0

        Parameters:
            index (int): index to insert.
            sublayer (Layer): sublayer to insert

        Examples:
            .. code-block:: python
                import paddle.fluid as fluid

                with fluid.dygraph.guard():
                    linears = fluid.dygraph.LayerList([fluid.dygraph.Linear(10, 10) for i in range(10)])
                    another = fluid.dygraph.Linear(10, 10)
                    linears.insert(3, another)
                    print(linears[3] is another)  # True
        """
        if len(self._sub_layers)==0:
            self.append(sublayer)
            return
        assert isinstance(index, int) and \
               0 <= index < len(self._sub_layers), \
            "index should be an integer in range [0, len(self))"
        for i in range(len(self._sub_layers), index, -1):
            self._sub_layers[str(i)] = self._sub_layers[str(i - 1)]
        self._sub_layers[str(index)] = sublayer





class ConstantPad2d(Module):
    def __init__(self, padding, value):
        super().__init__()
        self.value = value
        if isinstance(padding, int):
            self.padding = [padding] * 4
        else:
            self.padding = padding

    def forward(self, x):
        return fluid.layers.pad2d(x, self.padding, pad_value=self.value, mode='constant')


class ReplicationPad2d(Module):
    def __init__(self, padding):
        super().__init__()
        if isinstance(padding, int):
            self.padding = [padding] * 4
        else:
            self.padding = padding

    def forward(self, x):
        return fluid.layers.pad2d(x, self.padding, mode='edge')


class ReflectionPad2d(Module):
    def __init__(self, padding):
        super().__init__()
        if isinstance(padding, int):
            self.padding = [padding] * 4
        else:
            self.padding = padding

    def forward(self, x):
        return fluid.layers.pad2d(x, self.padding, mode='reflect')


def  L1Loss():
    return fluid.dygraph.L1Loss()

def  MSELoss():
    return fluid.dygraph.MSELoss()

class BCEWithLogitsLoss():
    def __init__(self, weight=None, reduction='mean'):
        self.weight = weight
        self.reduction = reduction

    def __call__(self, x, label):
        out =  fluid.layers.sigmoid_cross_entropy_with_logits(x, label)
        if self.reduction == 'sum':
            return convertTensor(fluid.layers.reduce_sum(out))
        elif self.reduction == 'mean':
            return convertTensor(fluid.layers.reduce_mean(out))
        else:
            return convertTensor(out)

class Spectralnorm(Module):
    def __init__(self,
                 layer,
                 dim=0,
                 power_iters=1,
                 eps=1e-12,
                 dtype='float32'):
        super(Spectralnorm, self).__init__()

        self.dim = dim
        self.power_iters = power_iters
        self.eps = eps
        self.layer = layer
        self.has_bias=False
        self.is_fc=False
        if 'bias' in layer._parameters:
            bias=layer._parameters['bias']
            self.bias_orig =  self.create_parameter(bias.shape, dtype=bias.dtype)
            self.bias_orig.set_value(bias)
            self.has_bias=True
            del layer._parameters['bias']

        weight = layer._parameters['weight']
        self.weight_orig =  self.create_parameter(weight.shape, dtype=weight.dtype)
        self.weight_orig.set_value(weight)

        if isinstance(layer,dygraph.Linear):
            self.is_fc=True
            self.spectral_norm = dygraph.SpectralNorm(layer.weight.shape[::-1], dim, power_iters, eps, dtype)
        else:
            self.spectral_norm =  dygraph.SpectralNorm(layer.weight.shape, dim, power_iters, eps, dtype)
        del layer._parameters['weight']



    def forward(self, x):

        if self.is_fc:
            weight = self.spectral_norm(fluid.layers.transpose(self.weight_orig, [1, 0]))
            weight = fluid.layers.transpose(weight, [1, 0])
        else:
            weight = self.spectral_norm( self.weight_orig )

        self.layer.weight = weight
        if self.has_bias:
            self.layer.bias=self.bias_orig
        out = self.layer(x)
        return out

def split_states(states, bidirectional=False, state_components=1):
    r"""
    Split states of RNN network into possibly nested list or tuple of
    states of each RNN cells of the RNN network.
    Parameters:
        states (Tensor|tuple|list): the concatenated states for RNN network.
            When `state_components` is 1, states in a Tensor with shape
            `(L*D, N, C)` where `L` is the number of layers of the RNN
            network, `D` is the number of directions of the RNN network(1
            for unidirectional RNNs and 2 for bidirectional RNNs), `N` is
            the batch size of the input to the RNN network, `C` is the
            hidden size of the RNN network.
            When `state_components` is larger than 1, `states` is a tuple of
            `state_components` Tensors that meet the requirements described
            above.

            For SimpleRNNs and GRUs, `state_components` is 1, and for LSTMs,
            `state_components` is 2.
        bidirectional (bool): whether the state is of a bidirectional RNN
            network. Defaults to False.
        state_components (int): the number of the components of the states. see
            `states` above. Defaults to 1.

    Returns:
        A nested list or tuple of RNN cell states.
        If `bidirectional` is True, it can be indexed twice to get an RNN
        cell state. The first index indicates the layer, the second index
        indicates the direction.
        If `bidirectional` is False, it can be indexed once to get an RNN
        cell state. The index indicates the layer.
        Note that if `state_components` is larger than 1, an RNN cell state
        can be indexed one more time to get a tensor of shape(N, C), where
        `N` is the batch size of the input to the RNN cell, and `C` is the
        hidden size of the RNN cell.
    """
    if state_components == 1:
        states = paddle.unstack(states)
        if not bidirectional:
            return states
        else:
            return list(zip(states[::2], states[1::2]))
    else:
        assert len(states) == state_components
        states = tuple([paddle.unstack(item) for item in states])
        if not bidirectional:
            return list(zip(*states))
        else:
            states = list(zip(*states))
            return list(zip(states[::2], states[1::2]))


def concat_states(states, bidirectional=False, state_components=1):
    r"""
    Concatenate a possibly nested list or tuple of RNN cell states into a
    compact form.
    Parameters:
        states (list|tuple): a possibly nested list or tuple of RNN cell
            states.
            If `bidirectional` is True, it can be indexed twice to get an
            RNN cell state. The first index indicates the layer, the second
            index indicates the direction.
            If `bidirectional` is False, it can be indexed once to get an RNN
            cell state. The index indicates the layer.
            Note that if `state_components` is larger than 1, an RNN cell
            state can be indexed one more time to get a tensor of shape(N, C),
            where `N` is the batch size of the input to the RNN cell, and
            `C` is the hidden size of the RNN cell.
        bidirectional (bool): whether the state is of a bidirectional RNN
            network. Defaults to False.
        state_components (int): the number of the components of the states. see
            `states` above. Defaults to 1.

    Returns:
        Concatenated states for RNN network.
        When `state_components` is 1, states in a Tensor with shape
        `(L\*D, N, C)` where `L` is the number of layers of the RNN
        network, `D` is the number of directions of the RNN network(1 for
        unidirectional RNNs and 2 for bidirectional RNNs), `N` is the batch
        size of the input to the RNN network, `C` is the hidden size of the
        RNN network.

    """
    if state_components == 1:
        return paddle.stack(flatten(states))
    else:
        states = flatten(states)
        componnets = []
        for i in range(state_components):
            componnets.append(states[i::state_components])
        return tuple([paddle.stack(item) for item in componnets])


class RNNCellBase(Module):
    r"""
    RNNCellBase is the base class for abstraction representing the calculations
    mapping the input and state to the output and new state. It is suitable to
    and mostly used in RNN.
    """

    def get_initial_states(self,
                           batch_ref,
                           shape=None,
                           dtype=None,
                           init_value=0.,
                           batch_dim_idx=0):
        r"""
        Generate initialized states according to provided shape, data type and
        value.
        Parameters:
            batch_ref (Tensor): A tensor, which shape would be used to
                determine the batch size, which is used to generate initial
                states. For `batch_ref`'s shape d, `d[batch_dim_idx]` is
                treated as batch size.
            shape (list|tuple, optional): A (possibly nested structure of) shape[s],
                where a shape is a list/tuple of integer. `-1` (for batch size)
                will be automatically prepended if a shape does not starts with
                it. If None, property `state_shape` will be used. Defaults to
                None.
            dtype (str|list|tuple, optional): A (possibly nested structure of)
                data type[s]. The structure must be same as that of `shape`,
                except when all tensors' in states has the same data type, a
                single data type can be used. If None and property `cell.state_shape`
                is not available, current default floating type of paddle is
                used. Defaults to None.
            init_value (float, optional): A float value used to initialize states.
                Defaults to 0.
            batch_dim_idx (int, optional): An integer indicating which
                dimension of the of `batch_ref` represents batch. Defaults to 0.

        Returns:
            init_states (Tensor|tuple|list): tensor of the provided shape and
                dtype, or list of tensors that each satisfies the requirements,
                packed in the same structure as `shape` and `type` does.
        """
        # TODO: use inputs and batch_size
        batch_ref = flatten(batch_ref)[0]

        def _is_shape_sequence(seq):
            if sys.version_info < (3,):
                integer_types = (
                    int,
                    "int32")
            else:
                integer_types = (int,)
            """For shape, list/tuple of integer is the finest-grained objection"""
            if (isinstance(seq, list) or isinstance(seq, tuple)):
                if reduce(lambda flag, x: isinstance(x, integer_types) and flag,
                          seq, True):
                    return False
            # TODO: Add check for the illegal
            if isinstance(seq, dict):
                return True
            return (isinstance(seq, collections.Sequence) and
                    not isinstance(seq, six.string_types))

        class Shape(object):
            def __init__(self, shape):
                self.shape = shape if shape[0] == -1 else ([-1] + list(shape))

        # nested structure of shapes
        states_shapes = self.state_shape if shape is None else shape
        is_sequence_ori = utils.is_sequence
        utils.is_sequence = _is_shape_sequence
        states_shapes = map_structure(lambda shape: Shape(shape), states_shapes)
        utils.is_sequence = is_sequence_ori

        # nested structure of dtypes
        try:
            states_dtypes = self.state_dtype if dtype is None else dtype
        except NotImplementedError:
            states_dtypes = framework.get_default_dtype()
        if len(flatten(states_dtypes)) == 1:
            dtype = flatten(states_dtypes)[0]
            states_dtypes = map_structure(lambda shape: dtype, states_shapes)

        init_states = map_structure(
            lambda shape, dtype: paddle.fluid.layers.fill_constant_batch_size_like(
                input=batch_ref,
                shape=shape.shape,
                dtype=dtype,
                value=init_value,
                input_dim_idx=batch_dim_idx), states_shapes, states_dtypes)
        return init_states

    @property
    def state_shape(self):
        r"""
        Abstract method (property).
        Used to initialize states.
        A (possiblely nested structure of) shape[s], where a shape is a
        list/tuple of integers (-1 for batch size would be automatically
        inserted into a shape if shape is not started with it).
        Not necessary to be implemented if states are not initialized by
        `get_initial_states` or the `shape` argument is provided when using
        `get_initial_states`.
        """
        raise NotImplementedError(
            "Please add implementaion for `state_shape` in the used cell.")

    @property
    def state_dtype(self):
        r"""
        Abstract method (property).
        Used to initialize states.
        A (possiblely nested structure of) data types[s]. The structure must be
        same as that of `shape`, except when all tensors' in states has the same
        data type, a signle data type can be used.
        Not necessary to be implemented if states are not initialized
        by `get_initial_states` or the `dtype` argument is provided when using
        `get_initial_states`.
        """
        raise NotImplementedError(
            "Please add implementaion for `state_dtype` in the used cell.")


class GRUCell( Module):
    r"""
    Gated Recurrent Unit (GRU) RNN cell. Given the inputs and previous states,
    it computes the outputs and updates states.
    The formula for GRU used is as follows:
    ..  math::
        r_{t} & = \sigma(W_{ir}x_{t} + b_{ir} + W_{hr}h_{t-1} + b_{hr})
        z_{t} & = \sigma(W_{iz}x_{t} + b_{iz} + W_{hz}h_{t-1} + b_{hz})
        \widetilde{h}_{t} & = \tanh(W_{ic}x_{t} + b_{ic} + r_{t} * (W_{hc}h_{t-1} + b_{hc}))
        h_{t} & = z_{t} * h_{t-1} + (1 - z_{t}) * \widetilde{h}_{t}
        y_{t} & = h_{t}

    where :math:`\sigma` is the sigmoid fucntion, and * is the elemetwise
    multiplication operator.
    Please refer to `An Empirical Exploration of Recurrent Network Architectures
    <http://proceedings.mlr.press/v37/jozefowicz15.pdf>`_ for more details.
    Parameters:
        input_size (int): The input size.
        hidden_size (int): The hidden size.
        weight_ih_attr(ParamAttr, optional): The parameter attribute for
            `weight_ih`. Default: None.
        weight_hh_attr(ParamAttr, optional): The parameter attribute for
            `weight_hh`. Default: None.
        bias_ih_attr (ParamAttr, optional): The parameter attribute for the
            `bias_ih`. Default: None.
        bias_hh_attr (ParamAttr, optional): The parameter attribute for the
            `bias_hh`. Default: None.
        name (str, optional): Name for the operation (optional, default is
            None). For more information, please refer to :ref:`api_guide_Name`.
    Variables:
        - **weight_ih** (Parameter): shape (3 * hidden_size, input_size), input to hidden weight, which corresponds to the concatenation of :math:`W_{ir}, W_{iz}, W_{ic}` in the formula.
        - **weight_hh** (Parameter): shape (3 * hidden_size, hidden_size), hidden to hidden weight, which corresponds to the concatenation of :math:`W_{hr}, W_{hz}, W_{hc}` in the formula.
        - **bias_ih** (Parameter): shape (3 * hidden_size, ), input to hidden bias, which corresponds to the concatenation of :math:`b_{ir}, b_{iz}, b_{ic}` in the formula.
        - **bias_hh** (Parameter): shape (3 * hidden_size, ), hidden to hidden bias, swhich corresponds to the concatenation of :math:`b_{hr}, b_{hz}, b_{hc}` in the formula.
    Inputs:
        - **inputs** (Tensor): A tensor with shape `[batch_size, input_size]`, corresponding to :math:`x_t` in the formula.
        - **states** (Tensor): A tensor with shape `[batch_size, hidden_size]`, corresponding to :math:`h_{t-1}` in the formula.
    Returns:
        - **outputs** (Tensor): shape `[batch_size, hidden_size]`, the output, corresponding to :math:`h_{t}` in the formula.
        - **states** (Tensor): shape `[batch_size, hidden_size]`, the new hidden state, corresponding to :math:`h_{t}` in the formula.

    Notes:
        All the weights and bias are initialized with `Uniform(-std, std)` by
        default. Where std = :math:`\frac{1}{\sqrt{hidden\_size}}`. For more
        information about parameter initialization, please refer to s:ref:`api_fluid_ParamAttr`.
    Examples:
        .. code-block:: python
            import paddle
            x = paddle.randn((4, 16))
            prev_h = paddle.randn((4, 32))
            cell = paddle.nn.GRUCell(16, 32)
            y, h = cell(x, prev_h)
            print(y.shape)
            print(h.shape)
            #[4,32]
            #[4,32]
    """

    def __init__(self,input_size, hidden_size, bias=True):
        # (
        #          input_size,
        #          hidden_size,
        #          weight_ih_attr=None,
        #          weight_hh_attr=None,
        #          bias_ih_attr=None,
        #          bias_hh_attr=None,
        #          name=None):
        weight_ih_attr = None
        weight_hh_attr = None
        if bias:
            bias_ih_attr,bias_hh_attr = None,None
        else:
            bias_ih_attr, bias_hh_attr = False, False
        super(GRUCell, self).__init__()
        std = 1.0 / math.sqrt(hidden_size)
        self.weight_ih = self.create_parameter(
            (3 * hidden_size, input_size),
            weight_ih_attr,
            default_initializer=I.Uniform(-std, std))
        self.weight_hh = self.create_parameter(
            (3 * hidden_size, hidden_size),
            weight_hh_attr,
            default_initializer=I.Uniform(-std, std))
        self.bias_ih = self.create_parameter(
            (3 * hidden_size,),
            bias_ih_attr,
            is_bias=True,
            default_initializer=I.Uniform(-std, std))
        self.bias_hh = self.create_parameter(
            (3 * hidden_size,),
            bias_hh_attr,
            is_bias=True,
            default_initializer=I.Uniform(-std, std))

        self.hidden_size = hidden_size
        self.input_size = input_size
        self._gate_activation = F.sigmoid
        self._activation = paddle.tanh

    def forward(self, inputs, states=None):
        if states is None:
            states = self.get_initial_states(inputs, self.state_shape)

        pre_hidden = states
        x_gates = paddle.matmul(inputs, self.weight_ih, transpose_y=True)
        if self.bias_ih is not None:
            x_gates = x_gates + self.bias_ih
        h_gates = paddle.matmul(pre_hidden, self.weight_hh, transpose_y=True)
        if self.bias_hh is not None:
            h_gates = h_gates + self.bias_hh

        x_r, x_z, x_c = paddle.split(x_gates, num_or_sections=3, axis=1)
        h_r, h_z, h_c = paddle.split(h_gates, num_or_sections=3, axis=1)

        r = self._gate_activation(x_r + h_r)
        z = self._gate_activation(x_z + h_z)
        c = self._activation(x_c + r * h_c)  # apply reset gate after mm
        h = (pre_hidden - c) * z + c

        return h

    @property
    def state_shape(self):
        r"""
        The `state_shape` of GRUCell is a shape `[hidden_size]` (-1 for batch
        size would be automatically inserted into shape). The shape corresponds
        to the shape of :math:`h_{t-1}`.
        """
        return (self.hidden_size,)

    def extra_repr(self):
        return '{input_size}, {hidden_size}'.format(**self.__dict__)

class LayerNorm(Module ):
    r"""
    :alias_main: paddle.nn.LayerNorm
	:alias: paddle.nn.LayerNorm,paddle.nn.layer.LayerNorm,paddle.nn.layer.norm.LayerNorm
	:old_api: paddle.fluid.dygraph.LayerNorm
    This interface is used to construct a callable object of the ``LayerNorm`` class.
    For more details, refer to code examples.
    It implements the function of the Layer Normalization Layer and can be applied to mini-batch input data.
    Refer to `Layer Normalization <https://arxiv.org/pdf/1607.06450v1.pdf>`_
    The formula is as follows:
    ..  math::
        \\mu & = \\frac{1}{H}\\sum_{i=1}^{H} x_i
        \\sigma & = \\sqrt{\\frac{1}{H}\sum_{i=1}^{H}{(x_i - \\mu)^2} + \\epsilon}
        y & = f(\\frac{g}{\\sigma}(x - \\mu) + b)
    - :math:`x`: the vector representation of the summed inputs to the neurons in that layer.
    - :math:`H`: the number of hidden units in a layers
    - :math:`\\epsilon`: the small value added to the variance to prevent division by zero.
    - :math:`g`: the trainable scale parameter.
    - :math:`b`: the trainable bias parameter.
    Parameters:
        normalized_shape(int|list|tuple): Input shape from an expected input of
            size :math:`[*, normalized_shape[0], normalized_shape[1], ..., normalized_shape[-1]]`.
            If it is a single integer, this module will normalize over the last dimension
            which is expected to be of that specific size.
        epsilon(float, optional): The small value added to the variance to prevent
            division by zero. Default: 1e-05.
        weight_attr(ParamAttr|bool, optional): The parameter attribute for the learnable
            gain :math:`g`. If False, weight is None. If is None, a default :code:`ParamAttr` would be added as scale. The
            :attr:`param_attr` is initialized as 1 if it is added. Default: None.
        bias_attr(ParamAttr|bool, optional): The parameter attribute for the learnable
            bias :math:`b`. If is False, bias is None. If is None, a default :code:`ParamAttr` would be added as bias. The
            :attr:`bias_attr` is initialized as 0 if it is added. Default: None.
        name(str, optional): Name for the LayerNorm, default is None. For more information, please refer to :ref:`api_guide_Name`..
    Shape:
        - x: 2-D, 3-D, 4-D or 5-D tensor.
        - output: same shape as input x.
    Returns:
        None
    Examples:
        .. code-block:: python
          import paddle
          import numpy as np
          np.random.seed(123)
          x_data = np.random.random(size=(2, 2, 2, 3)).astype('float32')
          x = paddle.to_tensor(x_data)
          layer_norm = paddle.nn.LayerNorm(x_data.shape[1:])
          layer_norm_out = layer_norm(x)
          print(layer_norm_out)
    """

    def __init__(self,
                 normalized_shape,
                 eps=1e-05,
                 weight_attr=None,
                 bias_attr=None,
                 elementwise_affine=True,
                 name=None):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = [normalized_shape]
        if not elementwise_affine:
            weight_attr=False
            bias_attr=False

        self._normalized_shape = list(normalized_shape)
        self._epsilon = eps
        self._weight_attr = weight_attr
        self._bias_attr = bias_attr
        param_shape = [np.prod(self._normalized_shape)]

        if weight_attr is False:
            self.weight = None
        else:
            self.weight = self.create_parameter(
                attr=self._weight_attr,
                shape=param_shape,
                default_initializer=Constant(1.0))

        if bias_attr is False:
            self.bias = None
        else:
            self.bias = self.create_parameter(
                attr=self._bias_attr, shape=param_shape, is_bias=True)

    def forward(self, input):
        return layer_norm(
            input,
            normalized_shape=self._normalized_shape,
            weight=self.weight,
            bias=self.bias,
            epsilon=self._epsilon)

    def extra_repr(self):
        return 'normalized_shape={}, epsilon={}'.format(self._normalized_shape,
                                                        self._epsilon)


class _BatchNormBase(paddle.nn.BatchNorm,Module ):
    def __init__(self,num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        if affine:
            param_attr=None
            bias_attr = None
        else:
            param_attr=False
            bias_attr = False
        self.track_running_stats=track_running_stats

        do_model_average_for_mean_and_var=track_running_stats
        super(_BatchNormBase,self).__init__(num_features,momentum=momentum,epsilon=eps,param_attr=param_attr,bias_attr=bias_attr,do_model_average_for_mean_and_var=do_model_average_for_mean_and_var)

    def reset_running_stats(self):
        # if self.track_running_stats:
        init.zeros_(self._mean)
        init.constant_(self._variance,1)

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)



class BatchNorm1d(_BatchNormBase):
    r"""
    Applies Batch Normalization over a 2D or 3D input (a mini-batch of 1D inputswith additional channel dimension) as described in the paper Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift .
    When use_global_stats = False, the :math:`\\mu_{\\beta}`
    and :math:`\\sigma_{\\beta}^{2}` are the statistics of one mini-batch.
    Calculated as follows:
    ..  math::
        \\mu_{\\beta} &\\gets \\frac{1}{m} \\sum_{i=1}^{m} x_i \\qquad &//\\
        \ mini-batch\ mean \\\\
        \\sigma_{\\beta}^{2} &\\gets \\frac{1}{m} \\sum_{i=1}^{m}(x_i - \\
        \\mu_{\\beta})^2 \\qquad &//\ mini-batch\ variance \\\\
    When use_global_stats = True, the :math:`\\mu_{\\beta}`
    and :math:`\\sigma_{\\beta}^{2}` are not the statistics of one mini-batch.
    They are global or running statistics (moving_mean and moving_variance). It usually got from the
    pre-trained model. Calculated as follows:
    .. math::
        moving\_mean = moving\_mean * momentum + \mu_{\beta} * (1. - momentum) \quad &// global mean \\
        moving\_variance = moving\_variance * momentum + \sigma_{\beta}^{2} * (1. - momentum) \quad &// global variance \\
    The normalization function formula is as follows:
    ..  math::
        \\hat{x_i} &\\gets \\frac{x_i - \\mu_\\beta} {\\sqrt{\\
        \\sigma_{\\beta}^{2} + \\epsilon}} \\qquad &//\ normalize \\\\
        y_i &\\gets \\gamma \\hat{x_i} + \\beta \\qquad &//\ scale\ and\ shift
    - :math:`\\epsilon` : add a smaller value to the variance to prevent division by zero
    - :math:`\\gamma` : trainable proportional parameter
    - :math:`\\beta` : trainable deviation parameter
    Parameters:
        num_features(int): Indicate the number of channels of the input ``Tensor``.
        epsilon(float, optional): The small value added to the variance to prevent division by zero. Default: 1e-5.
        momentum(float, optional): The value used for the moving_mean and moving_var computation. Default: 0.9.
        weight_attr(ParamAttr|bool, optional): The parameter attribute for Parameter `scale`
            of batch_norm. If it is set to None or one attribute of ParamAttr, batch_norm
            will create ParamAttr as weight_attr. If it is set to Fasle, the weight is not learnable.
            If the Initializer of the weight_attr is not set, the parameter is initialized with Xavier. Default: None.
        bias_attr(ParamAttr|bool, optional): The parameter attribute for the bias of batch_norm.
            If it is set to None or one attribute of ParamAttr, batch_norm
            will create ParamAttr as bias_attr. If it is set to Fasle, the weight is not learnable.
            If the Initializer of the bias_attr is not set, the bias is initialized zero. Default: None.
        data_format(str, optional): Specify the input data format, may be "NC", "NCL" or "NLC". Defalut "NCL".
        use_global_stats(bool|None, optional): Whether to use global mean and variance. If set to False, use the statistics of one mini-batch, if set to True, use the global statistics, if set to None, use global statistics in the test phase and use the statistics of one mini-batch in the training phase. Default: None.
        name(str, optional): Name for the BatchNorm, default is None. For more information, please refer to :ref:`api_guide_Name`..
    Shape:
        - x: 2-D or 3-D tensor with shape: (batch, num_features) or (batch, num_features, length) when data_format is "NC" or "NCL",
            (batch, length, num_features) when data_format is "NLC".
        - output: 3-D tensor with same shape as input x.
    Returns:
        None.

    Examples:
        .. code-block:: python
          import paddle
          import numpy as np
          np.random.seed(123)
          x_data = np.random.random(size=(2, 1, 3)).astype('float32')
          x = paddle.to_tensor(x_data)
          batch_norm = paddle.nn.BatchNorm1D(1)
          batch_norm_out = batch_norm(x)
          print(batch_norm_out)
    """

    def _check_data_format(self, input):
        if input == 'NCHW' or input == 'NC' or input == 'NCL':
            self._data_format = 'NCHW'
        elif input == "NHWC" or input == 'NLC':
            self._data_format = "NHWC"
        else:
            raise ValueError(
                'expected NC , NCL, NLC or None for data_format input')

    def _check_input_dim(self, input):
        if len(input.shape) != 2 and len(input.shape) != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'.format(
                len(input.shape)))


class BatchNorm2d(_BatchNormBase):
    r"""
    Applies Batch Normalization over a 4D input (a mini-batch of 2D inputswith additional channel dimension) as described in the paper Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift .
    When use_global_stats = False, the :math:`\\mu_{\\beta}`
    and :math:`\\sigma_{\\beta}^{2}` are the statistics of one mini-batch.
    Calculated as follows:
    ..  math::
        \\mu_{\\beta} &\\gets \\frac{1}{m} \\sum_{i=1}^{m} x_i \\qquad &//\\
        \ mini-batch\ mean \\\\
        \\sigma_{\\beta}^{2} &\\gets \\frac{1}{m} \\sum_{i=1}^{m}(x_i - \\
        \\mu_{\\beta})^2 \\qquad &//\ mini-batch\ variance \\\\
    When use_global_stats = True, the :math:`\\mu_{\\beta}`
    and :math:`\\sigma_{\\beta}^{2}` are not the statistics of one mini-batch.
    They are global or running statistics (moving_mean and moving_variance). It usually got from the
    pre-trained model. Calculated as follows:
    .. math::
        moving\_mean = moving\_mean * momentum + \mu_{\beta} * (1. - momentum) \quad &// global mean \\
        moving\_variance = moving\_variance * momentum + \sigma_{\beta}^{2} * (1. - momentum) \quad &// global variance \\
    The normalization function formula is as follows:
    ..  math::
        \\hat{x_i} &\\gets \\frac{x_i - \\mu_\\beta} {\\sqrt{\\
        \\sigma_{\\beta}^{2} + \\epsilon}} \\qquad &//\ normalize \\\\
        y_i &\\gets \\gamma \\hat{x_i} + \\beta \\qquad &//\ scale\ and\ shift
    - :math:`\\epsilon` : add a smaller value to the variance to prevent division by zero
    - :math:`\\gamma` : trainable proportional parameter
    - :math:`\\beta` : trainable deviation parameter
    Parameters:
        num_features(int): Indicate the number of channels of the input ``Tensor``.
        epsilon(float, optional): The small value added to the variance to prevent division by zero. Default: 1e-5.
        momentum(float, optional): The value used for the moving_mean and moving_var computation. Default: 0.9.
        weight_attr(ParamAttr|bool, optional): The parameter attribute for Parameter `scale`
            of batch_norm. If it is set to None or one attribute of ParamAttr, batch_norm
            will create ParamAttr as weight_attr. If it is set to Fasle, the weight is not learnable.
            If the Initializer of the weight_attr is not set, the parameter is initialized with Xavier. Default: None.
        bias_attr(ParamAttr|bool, optional): The parameter attribute for the bias of batch_norm.
            If it is set to None or one attribute of ParamAttr, batch_norm
            will create ParamAttr as bias_attr. If it is set to Fasle, the weight is not learnable.
            If the Initializer of the bias_attr is not set, the bias is initialized zero. Default: None.
        data_format(str, optional): Specify the input data format, the data format can be "NCHW" or "NHWC". Default: NCHW.
        use_global_stats(bool|None, optional): Whether to use global mean and variance. If set to False, use the statistics of one mini-batch, if set to True, use the global statistics, if set to None, use global statistics in the test phase and use the statistics of one mini-batch in the training phase. Default: None.
        name(str, optional): Name for the BatchNorm, default is None. For more information, please refer to :ref:`api_guide_Name`..
    Shape:
        - x: 4-D tensor with shape: (batch, num_features, height, weight) when data_format is "NCHW",
            or (batch, height, weight, num_features) when data_format is "NHWC".
        - output: 4-D tensor with same shape as input x.
    Returns:
        None
    Examples:
        .. code-block:: python
          import paddle
          import numpy as np
          np.random.seed(123)
          x_data = np.random.random(size=(2, 1, 2, 3)).astype('float32')
          x = paddle.to_tensor(x_data)
          batch_norm = paddle.nn.BatchNorm2D(1)
          batch_norm_out = batch_norm(x)
          print(batch_norm_out)
    """

    def _check_data_format(self, input):
        if input == 'NCHW':
            self._data_format = input
        elif input == "NHWC":
            self._data_format = input
        else:
            raise ValueError('expected NCHW or NHWC for data_format input')

    def _check_input_dim(self, input):
        if len(input.shape) != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(
                len(input.shape)))


class BatchNorm3d(_BatchNormBase):
    r"""
    Applies Batch Normalization over a 5D input (a mini-batch of 3D inputswith additional channel dimension) as described in the paper Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift .
    When use_global_stats = False, the :math:`\\mu_{\\beta}`
    and :math:`\\sigma_{\\beta}^{2}` are the statistics of one mini-batch.
    Calculated as follows:
    ..  math::
        \\mu_{\\beta} &\\gets \\frac{1}{m} \\sum_{i=1}^{m} x_i \\qquad &//\\
        \ mini-batch\ mean \\\\
        \\sigma_{\\beta}^{2} &\\gets \\frac{1}{m} \\sum_{i=1}^{m}(x_i - \\
        \\mu_{\\beta})^2 \\qquad &//\ mini-batch\ variance \\\\
    When use_global_stats = True, the :math:`\\mu_{\\beta}`
    and :math:`\\sigma_{\\beta}^{2}` are not the statistics of one mini-batch.
    They are global or running statistics (moving_mean and moving_variance). It usually got from the
    pre-trained model. Calculated as follows:
    .. math::
        moving\_mean = moving\_mean * momentum + \mu_{\beta} * (1. - momentum) \quad &// global mean \\
        moving\_variance = moving\_variance * momentum + \sigma_{\beta}^{2} * (1. - momentum) \quad &// global variance \\
    The normalization function formula is as follows:
    ..  math::
        \\hat{x_i} &\\gets \\frac{x_i - \\mu_\\beta} {\\sqrt{\\
        \\sigma_{\\beta}^{2} + \\epsilon}} \\qquad &//\ normalize \\\\
        y_i &\\gets \\gamma \\hat{x_i} + \\beta \\qquad &//\ scale\ and\ shift
    - :math:`\\epsilon` : add a smaller value to the variance to prevent division by zero
    - :math:`\\gamma` : trainable proportional parameter
    - :math:`\\beta` : trainable deviation parameter
    Parameters:
        num_features(int): Indicate the number of channels of the input ``Tensor``.
        epsilon(float, optional): The small value added to the variance to prevent division by zero. Default: 1e-5.
        momentum(float, optional): The value used for the moving_mean and moving_var computation. Default: 0.9.
        weight_attr(ParamAttr|bool, optional): The parameter attribute for Parameter `scale`
            of batch_norm. If it is set to None or one attribute of ParamAttr, batch_norm
            will create ParamAttr as weight_attr. If it is set to Fasle, the weight is not learnable.
            If the Initializer of the weight_attr is not set, the parameter is initialized with Xavier. Default: None.
        bias_attr(ParamAttr|bool, optional): The parameter attribute for the bias of batch_norm.
            If it is set to None or one attribute of ParamAttr, batch_norm
            will create ParamAttr as bias_attr. If it is set to Fasle, the weight is not learnable.
            If the Initializer of the bias_attr is not set, the bias is initialized zero. Default: None.
        data_format(str, optional): Specify the input data format, the data format can be "NCDHW" or "NDHWC. Default: NCDHW.
        use_global_stats(bool|None, optional): Whether to use global mean and variance. If set to False, use the statistics of one mini-batch, if set to True, use the global statistics, if set to None, use global statistics in the test phase and use the statistics of one mini-batch in the training phase. Default: None.
        name(str, optional): Name for the BatchNorm, default is None. For more information, please refer to :ref:`api_guide_Name`..
    Shape:
        - x: 5-D tensor with shape: (batch, num_features, dims, height, weight) when data_format is "NCDHW",
            or (batch, dims, height, weight, num_features) when data_format is "NDHWC".
        - output: 5-D tensor with same shape as input x.
    Returns:
        None
    Examples:
        .. code-block:: python
          import paddle
          import numpy as np
          np.random.seed(123)
          x_data = np.random.random(size=(2, 1, 2, 2, 3)).astype('float32')
          x = paddle.to_tensor(x_data)
          batch_norm = paddle.nn.BatchNorm3D(1)
          batch_norm_out = batch_norm(x)
          print(batch_norm_out)
    """

    def _check_data_format(self, input):
        if input == 'NCHW' or input == 'NCDHW':
            self._data_format = 'NCHW'
        elif input == "NHWC" or input == "NDHWC":
            self._data_format = 'NHWC'
        else:
            raise ValueError(
                'expected NCDHW, NDHWC or None for data_format input')

    def _check_input_dim(self, input):
        if len(input.shape) != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(
                len(input.shape)))


class Softplus(paddle.nn.Softplus,Module):
    r"""
    Softplus Activation
    .. math::
        Softplus(x) = \\frac{1}{beta} * \\log(1 + e^{beta * x}) \\\\
        \\text{For numerical stability, the implementation reverts to the linear function when: beta * x > threshold.}
    Parameters:
        beta (float, optional): The value of beta for Softplus. Default is 1
        threshold (float, optional): The value of threshold for Softplus. Default is 20
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.
    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.
    Examples:
        .. code-block:: python
            import paddle
            import numpy as np
            x = paddle.to_tensor(np.array([-0.4, -0.2, 0.1, 0.3]))
            m = paddle.nn.Softplus()
            out = m(x) # [0.513015, 0.598139, 0.744397, 0.854355]
    """

    def __init__(self, beta=1, threshold=20, name=None):
        super(Softplus, self).__init__()
        self._beta = beta
        self._threshold = threshold
        self._name = name

    def forward(self, x):
        return F.softplus(x, self._beta, self._threshold, self._name)

    def extra_repr(self):
        name_str = ', name={}'.format(self._name) if self._name else ''
        return 'beta={}, threshold={}{}'.format(self._beta, self._threshold,
                                                name_str)

class Sigmoid(paddle.nn.Sigmoid,Module):
    """
    this interface is used to construct a callable object of the ``Sigmoid`` class. This layer calcluate the `sigmoid` of input x.
    .. math::
        Sigmoid(x) = \\frac{1}{1 + e^{-x}}
    Parameters:
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
    Shape:
        x: N-D tensor, available dtype is float16, float32, float64.
    Returns:
        A callable object of Sigmoid.
    Examples:
        .. code-block:: python
          import paddle
          m = paddle.nn.Sigmoid()
          x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
          out = m(x) # [0.7310586, 0.880797, 0.95257413, 0.98201376]
    """

    def __init__(self, name=None):
        super(Sigmoid, self).__init__()
        self.name = name

    def forward(self, x):
        return F.sigmoid(x, self.name)

    def extra_repr(self):
        name_str = 'name={}'.format(self.name) if self.name else ''
        return name_str

