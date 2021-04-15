

import torch

torch.manual_seed(0)
a = torch.randn(70839, 64 )

b = torch.randn(64, 64, requires_grad=True)


print(torch.argmax(torch.matmul(a,b)))
import paddorch
import paddle
a2 =paddorch.Tensor(a.detach().cpu().numpy())

b2 = paddorch.Tensor(b.detach().cpu().numpy())

print(paddle.argmax(paddorch.matmul(a2,b2) ))

