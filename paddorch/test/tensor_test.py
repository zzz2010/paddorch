import numpy as np
import math
import paddle.fluid as fluid
import paddle_torch as torch


if __name__ == '__main__':
    place = fluid.CPUPlace()
    with fluid.dygraph.guard(place=place):
        x=torch.Tensor(np.random.rand(5,3))
        y=x.new_full((3,5),1)
        y.fill_(2)

        z=y.unsqueeze(1)
        z=z.squeeze(1)

        w=z.clone()
        w=w.clamp_(0,1)

        w=w.permute( 1,0 )
        u=w.repeat( 3,6 )

        print(u,u.size())

        a=fluid.core.VarBase(np.random.randn(5))
        b=torch.Tensor(a)
        print(b.shape)