from paddle import  fluid

def to_dlpack(tensor):
    dltensor = tensor._to_dlpack()
    return dltensor

def from_dlpack(dlpack):
    tensor_from_dlpack = fluid.core.from_dlpack(dlpack)
    return tensor_from_dlpack


