from paddle import fluid
import paddorch as torch
import numpy as np

place = fluid.CPUPlace()
with fluid.dygraph.guard(place=place):
    rnn = torch.nn.GRU(16, 32, 2,batch_first=True)

    x = torch.randn((4, 23, 16))
    prev_h = torch.randn((2, 4, 32))
    y, h = rnn(x, prev_h)

    print(y.shape)
    print(h.shape)



