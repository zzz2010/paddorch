import paddorch
from paddorch import index_copy_inplace_nograd

memory=paddorch.zeros((4,3))
k=paddorch.arange(0,6).view(2,3)
out_ids=paddorch.LongTensor([1,3])

index_copy_inplace_nograd(memory, 0, out_ids, k)
print("paddorch",memory)



import torch


memory=torch.zeros((4,3))
k=torch.arange(0,6).view(2,3).float()
out_ids=torch.LongTensor([1,3])

memory.index_copy_( 0, out_ids, k)
print("pytorch",memory)