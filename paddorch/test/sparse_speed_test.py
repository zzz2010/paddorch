

import numpy as np

down_sample=10
N_dim=70839//down_sample
N_dim2=N_dim
np.random.seed(0)
from scipy import sparse
from numpy import array
nnz=1620256//down_sample
I =  np.random.choice(np.arange(N_dim),nnz)
J = np.random.choice(np.arange(N_dim2),nnz)
V = np.random.randn(nnz ).astype("float").tolist()
# a = sparse.coo_matrix((V,(I,J)),shape=(N_dim,N_dim2))
b_np = np.random.randn(N_dim2, 64 ).astype("float32")




import paddorch
import paddle
import numpy as np
import torch
from paddle import fluid

#
device="cuda"
device="cpu"


if device=="cuda":
    a = torch.cuda.sparse.FloatTensor(torch.LongTensor(np.stack([I, J])).cuda(), torch.FloatTensor(V).cuda(), (N_dim2, N_dim))
else:
    place = fluid.CPUPlace()
    a=torch.sparse.FloatTensor(torch.LongTensor(np.stack([I,J]) ), torch.FloatTensor(V ) ,(N_dim2, N_dim)   )


b=torch.from_numpy(b_np).to(device)
b_param=torch.nn.Parameter(b)
b.requires_grad=True
a.requires_grad=True
import time
before=time.time()

for _ in range(6):
    if device == "cuda":
        y = torch.sparse.mm(a, b )
    else:
        y = torch.sparse.mm(a, b)

    b=torch.cat([b,y],dim=1)

y.sum().backward()

after=time.time()
print("time:",after-before)

# print("max grad",torch.max(a.grad))
if device=="cuda":
    import sys
    sys.exit()

with fluid.dygraph.guard(place=place):
    a=paddorch.sparse.FloatTensor(paddorch.LongTensor(np.stack([I,J]) ), paddorch.FloatTensor(V ) ,(N_dim2, N_dim))
    b=paddorch.from_numpy(b_np )
    b_param=paddorch.nn.Parameter(b)
    b.stop_gradient=False
    a.values.stop_gradient=False

    import time
    before=time.time()
    for _ in range(6):
        y = paddorch.sparse.mm(a, b )

        b=paddorch.cat([b,y],dim=1)
        break

    y.sum().backward()

    after=time.time()
    print("time:",after-before)

    print("max grad",np.max(a.values.gradient()))




# y2=paddorch.mm(paddorch.from_numpy(a_dense),b)
# print(y2)

