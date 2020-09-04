import paddorch
from paddorch.tensor import  new_full
from  paddle import fluid
# class Parameter(paddorch.Tensor):
#     def __init__(self):
#         fluid.create_random_int_lodtensor()
#         fluid.create_lod_tensor()
#         fluid.dygraph.layers.Layer.create_parameter()

def Parameter(shape,fill_value=None,requires_grad=True):
    if isinstance(shape,fluid.framework.core.VarBase):
        X=Parameter(shape.shape,0.0)
        fluid.layers.assign(shape,X)
    else:
        if isinstance(shape,int):
            shape=[shape]
        # return fluid.dygraph.layers.create_parameter(layer,shape,default_initializer=fluid.initializer.ConstantInitializer(value=fill_value))
        # return new_full(shape,fill_value)
        X= fluid.layers.create_parameter(
                        shape=shape,dtype="float32",
                        attr=fluid.ParamAttr(name=None, initializer=fluid.initializer.ConstantInitializer(value=fill_value)),
                        is_bias=False)
    if not requires_grad:
        X.stop_gradient=True

    return X



