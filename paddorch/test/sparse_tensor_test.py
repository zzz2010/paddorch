

import torch
N_dim=1300
torch.manual_seed(0)
a = torch.randn(2, N_dim).to_sparse().requires_grad_(False)
a_dense=a.to_dense().numpy()
b = torch.randn(N_dim, 2, requires_grad=True)



y = torch.sparse.mm(a, b)
print(y)
import paddorch
import paddle
import numpy as np

tid=paddorch.LongTensor(np.arange(2))
dd=paddorch.randn((3,4))
tid2=paddorch.Tensor(dd)
print("test",tid2)
a=paddorch.sparse.FloatTensor(paddorch.LongTensor(a._indices().detach().numpy()),paddorch.FloatTensor(a._values().detach().numpy()),(2, N_dim))
b=paddorch.from_numpy(b.detach().numpy())

y = paddorch.sparse.mm(a, b)
print(y)




y2=paddorch.mm(paddorch.from_numpy(a_dense),b)
print(y2)