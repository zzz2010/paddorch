import paddle.fluid as fluid
import paddorch
import paddle

_initialized=True
def is_available():
    try:
        fluid.CUDAPlace(0)
        return True
    except:
        return False

def manual_seed_all(seed):
    paddorch.manual_seed(seed)


def manual_seed(seed):
    paddorch.manual_seed(seed)

def set_device(device):
    return paddle.set_device(device)


def empty_cache():
    return