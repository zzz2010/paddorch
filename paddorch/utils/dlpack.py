from paddle import  fluid
import numpy as np
import paddorch
import paddle
from sys import platform
def to_dlpack(tensor):
    if  True: # "win" in platform:
        import nnabla as nn  ##pip install nnabla==1.18.0
        from nnabla.utils.dlpack import to_dlpack
        from nnabla.ext_utils import get_extension_context
        ctx = get_extension_context('cpu')
        nn.set_default_context(ctx)
        a = nn.NdArray.from_numpy_array(tensor.numpy())
        return to_dlpack(a)
    else:
        a_lodtensor=tensor.cpu().value().get_tensor()
        return  a_lodtensor._to_dlpack()


def from_dlpack(dlpack):
    tensor_from_dlpack = fluid.core.from_dlpack(dlpack)
    place=tensor_from_dlpack._place()
    if  True:# "win" in platform: # CPU env
        if "int32" in str(tensor_from_dlpack):
            return paddorch.convertTensor(paddle.to_tensor(np.array(tensor_from_dlpack),dtype="int32"))
        else:
            return paddorch.Tensor(paddle.to_tensor(np.array(tensor_from_dlpack)))
    else:
        with paddle.fluid.dygraph.guard(place=place):
            tensor_from_dlpack.__class__=paddle.fluid.LoDTensor
            ret= paddle.Tensor( tensor_from_dlpack)
            if "int32" in str(tensor_from_dlpack):
                ret=paddle.to_tensor(ret,dtype="int32")
            tensor_from_dlpack.__class__=paddle.fluid.core_avx.Tensor
        return ret


