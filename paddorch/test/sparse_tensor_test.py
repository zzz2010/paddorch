

import torch
torch.manual_seed(0)
a = torch.randn(2, 3).to_sparse().requires_grad_(False)
a_dense=a.to_dense().numpy()
b = torch.randn(3, 2, requires_grad=True)
print(a)

print(a_dense)

y = torch.sparse.mm(a, b)
print(y)
import paddorch
import paddle

a=paddorch.sparse.FloatTensor(paddorch.LongTensor(a._indices().detach().numpy()),paddorch.FloatTensor(a._values().detach().numpy()),(2,3))
b=paddorch.from_numpy(b.detach().numpy())

y = paddorch.sparse.mm(a, b)
print(y)
mask=y.astype("bool")
print(y[mask])
print(y.astype("bool").dtype==paddle.fluid.core.VarDesc.VarType.BOOL)
cc=paddle.masked_select(y,y.astype("bool") )
print(cc)
y2=paddorch.mm(paddorch.from_numpy(a_dense),b)
print(y2)