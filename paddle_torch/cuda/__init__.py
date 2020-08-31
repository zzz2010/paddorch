import paddle.fluid as fluid



def is_available():
    try:
        fluid.CUDAPlace(0)
        return True
    except:
        return False