import paddle.fluid as fluid
import paddorch


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