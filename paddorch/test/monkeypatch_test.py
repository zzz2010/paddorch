import paddorch
import paddle


paddle.Tensor.new_full=paddorch.Tensor.new_full
paddle.Tensor.dim=paddorch.Tensor.dim
paddle.Tensor.min=paddorch.Tensor.min
paddle.Tensor.sum=paddorch.Tensor.sum
paddle.Tensor.mean=paddorch.Tensor.mean
paddle.Tensor.max=paddorch.Tensor.max
paddle.Tensor.new=paddorch.Tensor.new
paddle.Tensor.scatter_add_=paddorch.Tensor.scatter_add_
paddle.Tensor.scatter_add=paddorch.Tensor.scatter_add
paddle.Tensor._fill_=paddorch.Tensor._fill_
paddle.Tensor.fill_=paddorch.Tensor.fill_
paddle.Tensor.unsqueeze=paddorch.Tensor.unsqueeze
paddle.Tensor.narrow=paddorch.Tensor.narrow
paddle.Tensor.squeeze=paddorch.Tensor.squeeze
paddle.Tensor.bmm=paddorch.Tensor.bmm
paddle.Tensor.sqrt=paddorch.Tensor.sqrt
paddle.Tensor.normal_=paddorch.Tensor.normal_
paddle.Tensor.random_=paddorch.Tensor.random_
paddle.Tensor.pow=paddorch.Tensor.pow
paddle.Tensor.clone=paddorch.Tensor.clone
paddle.Tensor.clamp_=paddorch.Tensor.clamp_
paddle.Tensor.float=paddorch.Tensor.float
paddle.Tensor.long=paddorch.Tensor.long
paddle.Tensor.add_=paddorch.Tensor.add_
paddle.Tensor.matmul=paddorch.Tensor.matmul
paddle.Tensor.norm=paddorch.Tensor.norm
paddle.Tensor.div_=paddorch.Tensor.div_
paddle.Tensor.expand=paddorch.Tensor.expand
paddle.Tensor.copy_=paddorch.Tensor.copy_
paddle.Tensor.mm=paddorch.Tensor.mm
paddle.Tensor.mul=paddorch.Tensor.mul
paddle.Tensor.mul_=paddorch.Tensor.mul_
paddle.Tensor.permute=paddorch.Tensor.permute
paddle.Tensor.transpose=paddorch.Tensor.transpose
paddle.Tensor.to=paddorch.Tensor.to
paddle.Tensor.type=paddorch.Tensor.type
paddle.Tensor.contiguous=paddorch.Tensor.contiguous
paddle.Tensor.flip=paddorch.Tensor.flip
paddle.Tensor.view=paddorch.Tensor.view
paddle.Tensor.repeat=paddorch.Tensor.repeat
paddle.Tensor.add=paddorch.Tensor.add
paddle.Tensor.item=paddorch.Tensor.item
paddle.Tensor.t=paddorch.Tensor.t
paddle.Tensor.reshape=paddorch.Tensor.reshape
paddle.Tensor.__setitem__=paddorch.Tensor.__setitem__
paddle.Tensor.__getitem__=paddorch.Tensor.__getitem__
paddle.Tensor.index_copy_=paddorch.Tensor.index_copy_
paddle.Tensor.index_copy=paddorch.Tensor.index_copy
paddle.Tensor.new_empty=paddorch.Tensor.new_empty
paddle.Tensor.view_as=paddorch.Tensor.view_as
paddle.Tensor.clamp=paddorch.Tensor.clamp
paddle.Tensor.requires_grad_=paddorch.Tensor.requires_grad_
paddle.Tensor.set_gradient=paddorch.Tensor.set_gradient
paddle.Tensor.backward=paddorch.Tensor.backward
paddle.Tensor.new_zeros=paddorch.Tensor.new_zeros
paddle.Tensor.new_ones=paddorch.Tensor.new_ones
paddle.Tensor.sort=paddorch.Tensor.sort
paddle.Tensor.index_select=paddorch.Tensor.index_select
paddle.Tensor.masked_fill_=paddorch.Tensor.masked_fill_
paddle.Tensor.argmax=paddorch.Tensor.argmax
paddle.Tensor.tolist=paddorch.Tensor.tolist
paddle.Tensor.uniform_=paddorch.Tensor.uniform_
paddle.Tensor.__getstate__=paddorch.Tensor.__getstate__
paddle.Tensor.__setstate__=paddorch.Tensor.__setstate__


import numpy as np
a=paddle.Tensor(np.random.rand(3,4))

print(a.view(-1))