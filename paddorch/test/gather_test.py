import numpy as np
import torch
import paddorch
import paddle
x=np.array([[[0.9427, 0.0364, 0.2587],
         [0.4433, 0.3639, 0.4383],
         [0.5494, 0.4386, 0.2218]],
        [[0.1443, 0.9749, 0.3620],
         [0.6472, 0.0879, 0.7137],
         [0.2322, 0.3581, 0.8765]]])

print(x.shape)
i=np.array([[[0, 2, 1, 1, 2], [0, 0, 1, 0, 0], [2, 2, 2, 0, 2]], [[2, 1, 2, 1, 1], [2, 1, 2, 0, 0], [0, 1, 2, 1, 1]]])

print(i.shape)
out=np.array([[[0.9427, 0.0364, 0.2218],
         [0.5494, 0.0364, 0.2218],
         [0.4433, 0.3639, 0.2218],
         [0.4433, 0.0364, 0.2587],
         [0.5494, 0.0364, 0.2218]],
        [[0.2322, 0.3581, 0.3620],
         [0.6472, 0.0879, 0.7137],
         [0.2322, 0.3581, 0.8765],
         [0.6472, 0.9749, 0.7137],
         [0.6472, 0.9749, 0.7137]]])
print(out.shape)
torch_out=torch.gather(torch.FloatTensor(x),dim=-1,index=torch.LongTensor(i))

print("torch out",torch_out)

ind=paddle.Tensor(i.astype("int64"))
ind=ind.flatten()

row=paddle.expand( paddle.reshape(paddle.arange(x.shape[0]), (i.shape[0],1,1)), (i.shape)).flatten()
print(row)
col=paddle.expand( paddle.reshape(paddle.arange(x.shape[1]), (1,i.shape[1],1)), (i.shape)).flatten()
print(col)

ind2=paddle.stack([row,col,ind]).transpose([1,0])
print(ind2.shape)

paddle_out=paddle.gather_nd(paddle.Tensor(x), ind2).reshape(i.shape)
print("paddle out",paddle_out)



paddorch_out=paddorch.gather(paddle.Tensor(x),dim=-1,index=paddle.Tensor(i.astype("int64")))
print("paddorch out",paddorch_out)

assert  np.max(np.abs(paddorch_out.numpy()-torch_out.cpu().numpy()))<1e-7, "result not match"