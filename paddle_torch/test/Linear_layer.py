from paddle import fluid
import torch
import numpy as np

dim=2
## same weight and same input
w = np.linspace(0, 1, dim * dim).reshape(dim, dim).astype("float32")
xx = np.linspace(0, 1, dim).reshape(1, -1).astype("float32")

print("numpy output:", np.mean(np.dot(xx,w) ))

pytorch_fc = torch.nn.Linear(dim, dim, bias=False)
pytorch_fc.weight.data = torch.Tensor(w.T)
print("torch fc output:", torch.mean(pytorch_fc(torch.Tensor(xx))).detach().numpy())

place = fluid.CPUPlace()
with fluid.dygraph.guard(place=place):
    paddle_fc=fluid.dygraph.Linear(dim,dim,bias_attr=False)
    paddle_fc.weight.set_value(w)
    print("paddle fc output:", fluid.layers.mean(paddle_fc(fluid.core.VarBase(xx))).detach().numpy())


