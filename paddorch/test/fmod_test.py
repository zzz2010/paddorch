

import torch

torch.manual_seed(0)
a = torch.LongTensor([8,8,8,8])

b= torch.LongTensor([1,2,3,5])


print(torch.fmod(a,b))
import paddorch as torch
import paddle
a = torch.LongTensor([8,8,8,8])

b= torch.LongTensor([1,2,3,5])

print(torch.fmod(a,b))

