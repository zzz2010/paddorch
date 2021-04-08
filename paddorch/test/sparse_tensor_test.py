

import torch
N_dim=100
N_dim2=N_dim
torch.manual_seed(0)
a = torch.randn(N_dim2, N_dim).to_sparse().requires_grad_(False)
a_dense=a.to_dense().numpy()
b = torch.randn(N_dim, N_dim2, requires_grad=True)



y = torch.sparse.mm(a, b)
print(y)
import paddorch
import paddle
import numpy as np

paddle.enable_static()

a=paddorch.sparse.FloatTensor(paddorch.LongTensor(a._indices().detach().numpy()),paddorch.FloatTensor(a._values().detach().numpy()),(N_dim2, N_dim))
b=paddorch.from_numpy(b.detach().numpy())
b_param=paddorch.nn.Parameter(b)
b.stop_gradient=False
a.values.stop_gradient=False
import time
before=time.time()
for _ in range(6):
    y = paddorch.sparse.mm(a, b,fast=False)

    b=paddorch.cat([b,y],dim=1)

y.sum().backward()

after=time.time()
print("time:",after-before)
print(y)
print("max grad",a.values.gradient())




# y2=paddorch.mm(paddorch.from_numpy(a_dense),b)
# print(y2)

