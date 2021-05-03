from paddorch import nn
from paddle.fluid.layers.nn import clip_by_norm
def spectral_norm(layer,dim=0,
                 power_iters=1,
                 eps=1e-12,
                 dtype='float32'):
    return nn.Spectralnorm(layer,dim,power_iters,eps,dtype)


