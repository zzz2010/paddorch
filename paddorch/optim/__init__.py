from paddle.fluid.optimizer import  AdamOptimizer
from paddle import fluid
import paddorch.optim.lr_scheduler
def Adam(params ,
                    lr=0.001,
                    betas=None,
                    weight_decay=None):
    if betas is None:
        betas=[0.9,0.999]
    if weight_decay is None:
        weight_decay=0.0
    return AdamOptimizer( learning_rate=lr,
                 beta1=betas[0],
                 beta2=betas[1],
                 epsilon=1e-8,
                 parameter_list=params,
                 regularization=fluid.regularizer.L2DecayRegularizer(
                regularization_coeff=weight_decay),
                 grad_clip=None,
                 name=None,
                 lazy_mode=False)