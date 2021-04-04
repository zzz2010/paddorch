import paddle.fluid as fluid
from paddle.fluid import dygraph
from paddle.fluid.dygraph import layers,Conv2D,Linear,InstanceNorm
from paddle.fluid.dygraph import Layer
from . import functional as F
from paddle.fluid.framework import Variable, in_dygraph_mode
from .parameter import Parameter
from paddle.fluid import core
from paddle.fluid.data_feeder import convert_dtype, check_variable_and_dtype, check_type, check_dtype
import paddorch.nn.utils
import paddle
from . import init
from ..tensor import Tensor
from paddle.nn import LogSigmoid


def clone_layer(layer):
    new_obj=Layer()
    for name, layer in layer._sub_layers.items():
        new_obj.add_sublayer(name, clone_layer(layer))
    return new_obj


def forward_post_hook(layer,input,output):
    if isinstance(output,tuple):
        return tuple([Tensor(x) if isinstance(x,dygraph.core.VarBase) else x for x in output])
    else:
        if isinstance(output,dygraph.core.VarBase) and not isinstance(output,Tensor):
            return Tensor(output)
        else:
            return output

def forward_pre_hook(layer,input):
    if isinstance(input,tuple):
        return tuple([Tensor(x) if isinstance(x,dygraph.core.VarBase) else x for x in input])
    else:
        if isinstance(input,dygraph.core.VarBase) and not isinstance(input,Tensor):
            return Tensor(input)
        else:
            return input

class Module(Layer):
    def __init__(self , name_scope=None, dtype=core.VarDesc.VarType.FP32):
        super(Module, self).__init__(name_scope,dtype)
        self.register_buffer=dict()

        self.register_forward_post_hook(forward_post_hook)
        self.register_forward_pre_hook(forward_pre_hook)

    def eval(self):
        super(Module, self).eval()
        return self
    def load_state_dict(self,new_dict, strict=True):
        self.set_dict(new_dict, include_sublayers=True, use_structured_name=True)

    def register_parameter(self,name,value):
        self.__setattr__(name,Parameter(value))

    def register_buffer(self,name,value):
        X=Parameter(value)
        X.stop_gradient=True
        self.__setattr__(name,X)
    def add_module(self,name,layer):
        return self.add_sublayer(name,layer)

    def modules(self):
        return self.sublayers()

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
        bias_attr = None
        if not bias:
            bias_attr = False
        else:
            bias_attr =fluid.initializer.MSRAInitializer() # fluid.initializer.ConstantInitializer(value=0)
        super(Linear, self).__init__(in_features, out_features, param_attr=fluid.initializer.MSRAInitializer(), bias_attr=bias_attr, act=None, dtype="float32")





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





class Embedding(dygraph.Embedding,Module):
    def __init__(self,num_embeddings: int, embedding_dim: int,
                 padding_idx  = None, max_norm = None, norm_type: float = 2.0, scale_grad_by_freq: bool = False,
                 sparse: bool = False, _weight = None):
        super(Embedding,self).__init__(size=[num_embeddings,embedding_dim],
                 is_sparse=sparse,
                 is_distributed=False,
                 padding_idx=padding_idx,
                 param_attr=None,
                 dtype='float32')
        self.norm_type=norm_type
        self.max_norm=max_norm


    def forward(self, input):
        if self.max_norm is not None:
            max_norm=paddorch.norm(self.weight, p=self.norm_type, keepdim=True)
            max_norm=paddorch.clamp(max_norm,0,self.max_norm)
            normalized_weight=self.weight / max_norm
            paddorch.copy(normalized_weight,self.weight )
        y=super(Embedding, self).forward(input)
        return y

def Dropout(p=0.5, inplace=False):
    return dygraph.Dropout(p,dropout_implementation='upscale_in_train')


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



def Sequential(*layers):
    return dygraph.Sequential(*layers)


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
            return Tensor(fluid.layers.reduce_sum(out))
        elif self.reduction == 'mean':
            return Tensor(fluid.layers.reduce_mean(out))
        else:
            return Tensor(out)

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
