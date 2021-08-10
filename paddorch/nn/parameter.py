import paddle.nn

import paddorch
from paddorch.tensor import  new_full
from  paddle import fluid
# class Parameter(paddorch.Tensor):
#     def __init__(self):
#         fluid.create_random_int_lodtensor()
#         fluid.create_lod_tensor()
#         fluid.dygraph.layers.Layer.create_parameter()

def Parameter(shape_or_tensor, fill_value=None, requires_grad=True):
    if isinstance(shape_or_tensor, paddle.Tensor):
        X=Parameter(shape_or_tensor.shape, 0.0)
        fluid.layers.assign(shape_or_tensor.astype("float32"), X)
    else:
        if isinstance(shape_or_tensor, int):
            shape_or_tensor=[shape_or_tensor]
        # return fluid.dygraph.layers.create_parameter(layer,shape,default_initializer=fluid.initializer.ConstantInitializer(value=fill_value))
        # return new_full(shape,fill_value)

        X= paddle.create_parameter(
                        shape=shape_or_tensor,dtype="float32",
                        attr=fluid.ParamAttr(name=None, initializer=fluid.initializer.ConstantInitializer(value=fill_value)),
                        is_bias=False)
    if not requires_grad:
        X.stop_gradient=True

    return X



