from paddle import  fluid
import numpy as np
import paddorch
import paddle
 
def to_dlpack(tensor):
    a_lodtensor=tensor.value().get_tensor()
    return  a_lodtensor._to_dlpack()


def from_dlpack(dlpack):
    tensor_from_dlpack = fluid.core.from_dlpack(dlpack)
    # return paddorch.Tensor(paddle.fluid.dygraph.to_variable(tensor_from_dlpack))
    if "int64" in str(tensor_from_dlpack):
        return paddorch.convertTensor(paddorch.Tensor(np.array(tensor_from_dlpack)).astype("int64"))
    else:
        return paddorch.Tensor(np.array(tensor_from_dlpack))


