from paddle import  fluid
import numpy as np
import paddorch
import paddle
 
def to_dlpack(tensor):
    a_lodtensor=tensor.value().get_tensor()
    return  a_lodtensor._to_dlpack()


def from_dlpack(dlpack):
    tensor_from_dlpack = fluid.core.from_dlpack(dlpack)
    place=tensor_from_dlpack._place()
    # print(ret)
    # with paddle.fluid.dygraph.guard(place=place):
    tensor_from_dlpack.__class__=paddle.fluid.LoDTensor
    ret= paddle.Tensor( tensor_from_dlpack)
    tensor_from_dlpack.__class__=paddle.fluid.core_avx.Tensor
    ret= paddorch.convertTensor(ret)

    # 
    # if "cpu" in str(place).lower():
    #     if "int64" in str(tensor_from_dlpack):
    #         ret= paddorch.convertTensor(paddorch.Tensor(np.array(tensor_from_dlpack)).astype("int64"))
    #     else:
    #         ret= paddorch.Tensor(np.array(tensor_from_dlpack))
    # else:
    #     tensor_from_dlpack.__class__=paddle.fluid.LoDTensor
    #     ret= paddle.Tensor( tensor_from_dlpack)
    #     tensor_from_dlpack.__class__=paddle.fluid.core_avx.Tensor
    #     ret= paddorch.convertTensor(ret)
    #     print(ret)

    # print(place,ret.place)
    return ret


