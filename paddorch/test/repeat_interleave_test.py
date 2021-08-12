import paddle
import paddorch as paddorch
import torch
import numpy as np

x=np.arange(2*768).reshape((1,2,768))
x_t=torch.repeat_interleave(torch.FloatTensor(x),repeats=4,dim=1)
x_p=paddorch.repeat_interleave(paddorch.FloatTensor(x),repeats=4,dim=1)

assert np.max(np.abs(x_t.numpy()-x_p.numpy()))<0.001, "fail match torch"
print(paddorch.repeat_interleave(x_p,2))

y = paddorch.tensor([[1, 2], [3, 4]])

print(paddorch.repeat_interleave(y, 2))

print(paddorch.repeat_interleave(y, 3, dim=1))
