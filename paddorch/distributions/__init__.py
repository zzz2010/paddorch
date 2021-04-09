from paddle.fluid.layers import control_flow
from paddle.fluid.layers import tensor
from paddle.fluid.layers import ops
from paddle.fluid.layers import nn
from paddle.fluid.layers import elementwise_mul, elementwise_div, elementwise_add, elementwise_sub
from paddle.fluid import core
from paddle.fluid.framework import in_dygraph_mode
from paddle.tensor import arange, gather_nd, concat, multinomial
import math
import numpy as np
import warnings

from paddle.fluid.data_feeder import convert_dtype, check_variable_and_dtype, check_type, check_dtype


class Distribution(object):
    """
    The abstract base class for probability distributions. Functions are
    implemented in specific distributions.
    """

    def __init__(self):
        super(Distribution, self).__init__()

    def sample(self):
        """Sampling from the distribution."""
        raise NotImplementedError

    def entropy(self):
        """The entropy of the distribution."""
        raise NotImplementedError

    def kl_divergence(self, other):
        """The KL-divergence between self distributions and other."""
        raise NotImplementedError

    def log_prob(self, value):
        """Log probability density/mass function."""
        raise NotImplementedError

    def probs(self, value):
        """Probability density/mass function."""
        raise NotImplementedError

    def _validate_args(self, *args):
        """
        Argument validation for distribution args
        Args:
            value (float, list, numpy.ndarray, Tensor)
        Raises
            ValueError: if one argument is Tensor, all arguments should be Tensor
        """
        is_variable = False
        is_number = False
        for arg in args:
            if isinstance(arg, tensor.Variable):
                is_variable = True
            else:
                is_number = True

        if is_variable and is_number:
            raise ValueError(
                'if one argument is Tensor, all arguments should be Tensor')

        return is_variable

    def _to_tensor(self, *args):
        """
        Argument convert args to Tensor
        Args:
            value (float, list, numpy.ndarray, Tensor)
        Returns:
            Tensor of args.
        """
        numpy_args = []
        variable_args = []
        tmp = 0.

        for arg in args:
            if isinstance(arg, float):
                arg = [arg]
            if not isinstance(arg, (list, np.ndarray, tensor.Variable)):
                raise TypeError(
                    "Type of input args must be float, list, numpy.ndarray or Tensor, but received type {}".
                    format(type(arg)))

            arg_np = np.array(arg)
            arg_dtype = arg_np.dtype
            if str(arg_dtype) != 'float32':
                if str(arg_dtype) != 'float64':
                    # "assign" op doesn't support float64. if dtype is float64, float32 variable will be generated
                    #  and converted to float64 later using "cast".
                    warnings.warn(
                        "data type of argument only support float32 and float64, your argument will be convert to float32."
                    )
                arg_np = arg_np.astype('float32')
            # tmp is used to support broadcast, it summarizes shapes of all the args and get the mixed shape.
            tmp = tmp + arg_np
            numpy_args.append(arg_np)

        dtype = tmp.dtype
        for arg in numpy_args:
            arg_broadcasted, _ = np.broadcast_arrays(arg, tmp)
            arg_variable = tensor.create_tensor(dtype=dtype)
            tensor.assign(arg_broadcasted, arg_variable)
            variable_args.append(arg_variable)

        return tuple(variable_args)

    def _check_values_dtype_in_probs(self, param, value):
        """
        Log_prob and probs methods have input ``value``, if value's dtype is different from param,
        convert value's dtype to be consistent with param's dtype.
        Args:
            param (Tensor): low and high in Uniform class, loc and scale in Normal class.
            value (Tensor): The input tensor.
        Returns:
            value (Tensor): Change value's dtype if value's dtype is different from param.
        """
        if in_dygraph_mode():
            if value.dtype != param.dtype and convert_dtype(
                    value.dtype) in ['float32', 'float64']:
                warnings.warn(
                    "dtype of input 'value' needs to be the same as parameters of distribution class. dtype of 'value' will be converted."
                )
                return core.ops.cast(value, 'in_dtype', value.dtype,
                                     'out_dtype', param.dtype)
            return value

        check_variable_and_dtype(value, 'value', ['float32', 'float64'],
                                 'log_prob')
        if value.dtype != param.dtype:
            warnings.warn(
                "dtype of input 'value' needs to be the same as parameters of distribution class. dtype of 'value' will be converted."
            )
            return tensor.cast(value, dtype=param.dtype)
        return value

class Categorical(Distribution):
    r"""
    Categorical distribution is a discrete probability distribution that
    describes the possible results of a random variable that can take on
    one of K possible categories, with the probability of each category
    separately specified.
    The probability mass function (pmf) is:
    .. math::
        pmf(k; p_i) = \prod_{i=1}^{k} p_i^{[x=i]}
    In the above equation:
    * :math:`[x=i]` : it evaluates to 1 if :math:`x==i` , 0 otherwise.
    Args:
        logits(list|numpy.ndarray|Tensor): The logits input of categorical distribution. The data type is float32 or float64.
        name(str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
    Examples:
        .. code-block:: python
            import paddle
            from paddle.distribution import Categorical
            paddle.seed(100) # on CPU device
            x = paddle.rand([6])
            print(x)
            # [0.5535528  0.20714243 0.01162981
            #  0.51577556 0.36369765 0.2609165 ]
            paddle.seed(200) # on CPU device
            y = paddle.rand([6])
            print(y)
            # [0.77663314 0.90824795 0.15685187
            #  0.04279523 0.34468332 0.7955718 ]
            cat = Categorical(x)
            cat2 = Categorical(y)
            paddle.seed(1000) # on CPU device
            cat.sample([2,3])
            # [[0, 0, 5],
            #  [3, 4, 5]]
            cat.entropy()
            # [1.77528]
            cat.kl_divergence(cat2)
            # [0.071952]
            value = paddle.to_tensor([2,1,3])
            cat.probs(value)
            # [0.00608027 0.108298 0.269656]
            cat.log_prob(value)
            # [-5.10271 -2.22287 -1.31061]
    """

    def __init__(self, logits, name=None):
        """
        Args:
            logits(list|numpy.ndarray|Tensor): The logits input of categorical distribution. The data type is float32 or float64.
            name(str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        """
        if not in_dygraph_mode():
            check_type(logits, 'logits', (np.ndarray, tensor.Variable, list),
                       'Categorical')

        self.name = name if name is not None else 'Categorical'
        self.dtype = 'float32'

        if self._validate_args(logits):
            self.logits = logits
            self.dtype = convert_dtype(logits.dtype)
        else:
            if isinstance(logits, np.ndarray) and str(
                    logits.dtype) in ['float32', 'float64']:
                self.dtype = logits.dtype
            self.logits = self._to_tensor(logits)[0]
            if self.dtype != convert_dtype(self.logits.dtype):
                self.logits = tensor.cast(self.logits, dtype=self.dtype)

    def sample(self, shape):
        """Generate samples of the specified shape.
        Args:
            shape (list): Shape of the generated samples.
        Returns:
            Tensor: A tensor with prepended dimensions shape.

        Examples:
            .. code-block:: python
                import paddle
                from paddle.distribution import Categorical
                paddle.seed(100) # on CPU device
                x = paddle.rand([6])
                print(x)
                # [0.5535528  0.20714243 0.01162981
                #  0.51577556 0.36369765 0.2609165 ]
                cat = Categorical(x)
                paddle.seed(1000) # on CPU device
                cat.sample([2,3])
                # [[0, 0, 5],
                #  [3, 4, 5]]
        """
        name = self.name + '_sample'
        if not in_dygraph_mode():
            check_type(shape, 'shape', (list), 'sample')

        num_samples = np.prod(np.array(shape))

        logits_shape = list(self.logits.shape)
        if len(logits_shape) > 1:
            sample_shape = shape + logits_shape[:-1]
            logits = nn.reshape(self.logits,
                                [np.prod(logits_shape[:-1]), logits_shape[-1]])
        else:
            sample_shape = shape
            logits = self.logits

        sample_index = multinomial(logits, num_samples, True)
        return nn.reshape(sample_index, sample_shape, name=name)

    def kl_divergence(self, other):
        """The KL-divergence between two Categorical distributions.
        Args:
            other (Categorical): instance of Categorical. The data type is float32.
        Returns:
            Tensor: kl-divergence between two Categorical distributions.

        Examples:
            .. code-block:: python
                import paddle
                from paddle.distribution import Categorical
                paddle.seed(100) # on CPU device
                x = paddle.rand([6])
                print(x)
                # [0.5535528  0.20714243 0.01162981
                #  0.51577556 0.36369765 0.2609165 ]
                paddle.seed(200) # on CPU device
                y = paddle.rand([6])
                print(y)
                # [0.77663314 0.90824795 0.15685187
                #  0.04279523 0.34468332 0.7955718 ]
                cat = Categorical(x)
                cat2 = Categorical(y)
                cat.kl_divergence(cat2)
                # [0.071952]
        """
        name = self.name + '_kl_divergence'
        if not in_dygraph_mode():
            check_type(other, 'other', Categorical, 'kl_divergence')

        logits = self.logits - nn.reduce_max(self.logits, dim=-1, keep_dim=True)
        other_logits = other.logits - nn.reduce_max(
            other.logits, dim=-1, keep_dim=True)
        e_logits = ops.exp(logits)
        other_e_logits = ops.exp(other_logits)
        z = nn.reduce_sum(e_logits, dim=-1, keep_dim=True)
        other_z = nn.reduce_sum(other_e_logits, dim=-1, keep_dim=True)
        prob = e_logits / z
        kl = nn.reduce_sum(
            prob * (logits - nn.log(z) - other_logits + nn.log(other_z)),
            dim=-1,
            keep_dim=True,
            name=name)

        return kl

    def entropy(self):
        """Shannon entropy in nats.
        Returns:
            Tensor: Shannon entropy of Categorical distribution. The data type is float32.

        Examples:
            .. code-block:: python
                import paddle
                from paddle.distribution import Categorical
                paddle.seed(100) # on CPU device
                x = paddle.rand([6])
                print(x)
                # [0.5535528  0.20714243 0.01162981
                #  0.51577556 0.36369765 0.2609165 ]
                cat = Categorical(x)
                cat.entropy()
                # [1.77528]
        """
        name = self.name + '_entropy'
        logits = self.logits - nn.reduce_max(self.logits, dim=-1, keep_dim=True)
        e_logits = ops.exp(logits)
        z = nn.reduce_sum(e_logits, dim=-1, keep_dim=True)
        prob = e_logits / z

        neg_entropy = nn.reduce_sum(
            prob * (logits - nn.log(z)), dim=-1, keep_dim=True)
        entropy = nn.scale(neg_entropy, scale=-1.0, name=name)
        return entropy

    def probs(self, value):
        """Probabilities of the given category (``value``).
        If ``logits`` is 2-D or higher dimension, the last dimension will be regarded as
        category, and the others represents the different distributions.
        At the same time, if ``vlaue`` is 1-D Tensor, ``value`` will be broadcast to the
        same number of distributions as ``logits``.
        If ``value`` is not 1-D Tensor, ``value`` should have the same number distributions
        with ``logits. That is, ``value[:-1] = logits[:-1]``.
        Args:
            value (Tensor): The input tensor represents the selected category index.
        Returns:
            Tensor: probability according to the category index.

        Examples:
            .. code-block:: python
                import paddle
                from paddle.distribution import Categorical
                paddle.seed(100) # on CPU device
                x = paddle.rand([6])
                print(x)
                # [0.5535528  0.20714243 0.01162981
                #  0.51577556 0.36369765 0.2609165 ]
                cat = Categorical(x)
                value = paddle.to_tensor([2,1,3])
                cat.probs(value)
                # [0.00608027 0.108298 0.269656]
        """
        name = self.name + '_probs'

        dist_sum = nn.reduce_sum(self.logits, dim=-1, keep_dim=True)
        prob = self.logits / dist_sum

        shape = list(prob.shape)
        value_shape = list(value.shape)
        if len(shape) == 1:
            num_value_in_one_dist = np.prod(value_shape)
            index_value = nn.reshape(value, [num_value_in_one_dist, 1])
            index = index_value
        else:
            num_dist = np.prod(shape[:-1])
            num_value_in_one_dist = value_shape[-1]
            prob = nn.reshape(prob, [num_dist, shape[-1]])
            if len(value_shape) == 1:
                value = nn.expand(value, [num_dist])
                value_shape = shape[:-1] + value_shape
            index_value = nn.reshape(value, [num_dist, -1, 1])
            if shape[:-1] != value_shape[:-1]:
                raise ValueError(
                    "shape of value {} must match shape of logits {}".format(
                        str(value_shape[:-1]), str(shape[:-1])))

            index_prefix = nn.unsqueeze(
                arange(
                    num_dist, dtype=index_value.dtype), axes=-1)
            index_prefix = nn.expand(index_prefix, [1, num_value_in_one_dist])
            index_prefix = nn.unsqueeze(index_prefix, axes=-1)

            if index_value.dtype != index_prefix.dtype:
                tensor.cast(index_prefix, dtype=index_value.dtype)
            index = concat([index_prefix, index_value], axis=-1)

        # value is the category index to search for the corresponding probability.
        select_prob = gather_nd(prob, index)
        return nn.reshape(select_prob, value_shape, name=name)

    def log_prob(self, value):
        """Log probabilities of the given category. Refer to ``probs`` method.
        Args:
            value (Tensor): The input tensor represents the selected category index.
        Returns:
            Tensor: Log probability.

        Examples:
            .. code-block:: python
                import paddle
                from paddle.distribution import Categorical
                paddle.seed(100) # on CPU device
                x = paddle.rand([6])
                print(x)
                # [0.5535528  0.20714243 0.01162981
                #  0.51577556 0.36369765 0.2609165 ]
                cat = Categorical(x)
                value = paddle.to_tensor([2,1,3])
                cat.log_prob(value)
                # [-5.10271 -2.22287 -1.31061]
        """
        name = self.name + '_log_prob'

        return nn.log(self.probs(value), name=name)