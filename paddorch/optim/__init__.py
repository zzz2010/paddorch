from paddle.fluid.optimizer import  AdamOptimizer
from paddle import fluid
from . import lr_scheduler
from paddle.fluid.optimizer import Optimizer

import paddle

class Adam(paddle.optimizer.Adam):
    def __init__(self,params ,
                    lr=0.001,
                    betas=None,
                    weight_decay=None,eps=1e-8,grad_clip=None,lazy_mode=False):

        if betas is None:
            betas = [0.9, 0.999]

        super(Adam, self).__init__(learning_rate=lr,
                 beta1=betas[0],
                 beta2=betas[1],
                 epsilon=eps,
                 parameters=params,
                weight_decay=weight_decay ,
                 grad_clip=grad_clip,
                 name=None,
                 lazy_mode=lazy_mode)
    def zero_grad(self):
        self.clear_gradients()
