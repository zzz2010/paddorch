import paddorch
import paddle

#
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
# paddle.Tensor.type=paddorch.Tensor.type
paddle.Tensor.contiguous=paddorch.Tensor.contiguous
paddle.Tensor.flip=paddorch.Tensor.flip
paddle.Tensor.view=paddorch.Tensor.view
paddle.Tensor.repeat=paddorch.Tensor.repeat
paddle.Tensor.add=paddorch.Tensor.add
paddle.Tensor.item=paddorch.Tensor.item
paddle.Tensor.t=paddorch.Tensor.t
paddle.Tensor.reshape=paddorch.Tensor.reshape
paddle.Tensor.shape_orig=paddle.Tensor.shape
paddle.Tensor.shape=paddorch.Tensor.shape
# paddle.Tensor.__setitem__=paddorch.Tensor.__setitem__
# paddle.Tensor.__getitem__origin=paddle.Tensor.__getitem__
# paddle.Tensor.__getitem__=paddorch.Tensor.__getitem__
paddle.Tensor.index_copy_=paddorch.Tensor.index_copy_
paddle.Tensor.index_copy=paddorch.Tensor.index_copy
paddle.Tensor.new_empty=paddorch.Tensor.new_empty
paddle.Tensor.view_as=paddorch.Tensor.view_as
paddle.Tensor.clamp=paddorch.Tensor.clamp
paddle.Tensor.requires_grad_=paddorch.Tensor.requires_grad_
paddle.Tensor.set_gradient=paddorch.Tensor.set_gradient
paddle.Tensor.grad_orig=paddle.Tensor.grad
paddle.Tensor.grad=paddorch.Tensor.grad
paddle.Tensor.backward_orig=paddle.Tensor.backward
paddle.Tensor.backward=paddorch.Tensor.backward
paddle.Tensor.new_zeros=paddorch.Tensor.new_zeros
paddle.Tensor.new_ones=paddorch.Tensor.new_ones
paddle.Tensor.sort=paddorch.Tensor.sort
paddle.Tensor.index_select=paddorch.Tensor.index_select
paddle.Tensor.masked_fill_=paddorch.Tensor.masked_fill_
paddle.Tensor.masked_fill=paddorch.Tensor.masked_fill
paddle.Tensor.argmax=paddorch.Tensor.argmax
paddle.Tensor.tolist=paddorch.Tensor.tolist
paddle.Tensor.uniform_=paddorch.Tensor.uniform_
paddle.Tensor.__getstate__=paddorch.Tensor.__getstate__
paddle.Tensor.__setstate__=paddorch.Tensor.__setstate__
paddle.Tensor.byte=paddorch.Tensor.byte
paddle.Tensor.bernoulli_=paddorch.Tensor.bernoulli_
paddle.Tensor.bool=paddorch.Tensor.bool
paddle.Tensor.chunk=paddorch.Tensor.chunk
paddle.Tensor.__invert__=paddorch.Tensor.__invert__
paddle.Tensor.split=paddorch.Tensor.split
paddle.Tensor.device=paddorch.Tensor.device
paddle.Tensor.type_as=paddorch.Tensor.type_as
paddle.Tensor.new_tensor=paddorch.Tensor.new_tensor
paddle.Tensor.__or__=paddorch.Tensor.__or__
paddle.Tensor.ne=paddorch.Tensor.ne
paddle.Tensor.int=paddorch.Tensor.int
paddle.Tensor.triu=paddorch.Tensor.triu
paddle.Tensor.is_cuda=paddorch.Tensor.is_cuda
paddle.Tensor.fill_diagonal=paddorch.Tensor.fill_diagonal



# from functools import partial
# def monkeypatch(paddle_tensor):
#     # paddle_tensor.new_full = partial(paddorch.Tensor.new_full, paddle_tensor)
#     paddle_tensor.dim = partial(paddorch.Tensor.dim, paddle_tensor)
#     paddle_tensor.min = partial(paddorch.Tensor.min, paddle_tensor)
#     paddle_tensor.sum = partial(paddorch.Tensor.sum, paddle_tensor)
#     paddle_tensor.mean = partial(paddorch.Tensor.mean, paddle_tensor)
#     paddle_tensor.max = partial(paddorch.Tensor.max, paddle_tensor)
#     paddle_tensor.new = partial(paddorch.Tensor.new, paddle_tensor)
#     paddle_tensor.scatter_add_ = partial(paddorch.Tensor.scatter_add_, paddle_tensor)
#     paddle_tensor.scatter_add = partial(paddorch.Tensor.scatter_add, paddle_tensor)
#     paddle_tensor._fill_ = partial(paddorch.Tensor._fill_, paddle_tensor)
#     paddle_tensor.fill_ = partial(paddorch.Tensor.fill_, paddle_tensor)
#     paddle_tensor.unsqueeze = partial(paddorch.Tensor.unsqueeze, paddle_tensor)
#     paddle_tensor.narrow = partial(paddorch.Tensor.narrow, paddle_tensor)
#     paddle_tensor.squeeze = partial(paddorch.Tensor.squeeze, paddle_tensor)
#     paddle_tensor.bmm = partial(paddorch.Tensor.bmm, paddle_tensor)
#     paddle_tensor.sqrt = partial(paddorch.Tensor.sqrt, paddle_tensor)
#     paddle_tensor.normal_ = partial(paddorch.Tensor.normal_, paddle_tensor)
#     paddle_tensor.random_ = partial(paddorch.Tensor.random_, paddle_tensor)
#     paddle_tensor.pow = partial(paddorch.Tensor.pow, paddle_tensor)
#     paddle_tensor.clone = partial(paddorch.Tensor.clone, paddle_tensor)
#     paddle_tensor.clamp_ = partial(paddorch.Tensor.clamp_, paddle_tensor)
#     paddle_tensor.float = partial(paddorch.Tensor.float, paddle_tensor)
#     paddle_tensor.long = partial(paddorch.Tensor.long, paddle_tensor)
#     paddle_tensor.add_ = partial(paddorch.Tensor.add_, paddle_tensor)
#     paddle_tensor.matmul = partial(paddorch.Tensor.matmul, paddle_tensor)
#     paddle_tensor.norm = partial(paddorch.Tensor.norm, paddle_tensor)
#     paddle_tensor.div_ = partial(paddorch.Tensor.div_, paddle_tensor)
#     paddle_tensor.expand = partial(paddorch.Tensor.expand, paddle_tensor)
#     paddle_tensor.copy_ = partial(paddorch.Tensor.copy_, paddle_tensor)
#     paddle_tensor.mm = partial(paddorch.Tensor.mm, paddle_tensor)
#     paddle_tensor.mul = partial(paddorch.Tensor.mul, paddle_tensor)
#     paddle_tensor.mul_ = partial(paddorch.Tensor.mul_, paddle_tensor)
#     paddle_tensor.permute = partial(paddorch.Tensor.permute, paddle_tensor)
#     paddle_tensor.transpose = partial(paddorch.Tensor.transpose, paddle_tensor)
#     paddle_tensor.to = partial(paddorch.Tensor.to, paddle_tensor)
#     paddle_tensor.type = partial(paddorch.Tensor.type, paddle_tensor)
#     paddle_tensor.contiguous = partial(paddorch.Tensor.contiguous, paddle_tensor)
#     paddle_tensor.flip = partial(paddorch.Tensor.flip, paddle_tensor)
#     paddle_tensor.view = partial(paddorch.Tensor.view, paddle_tensor)
#     paddle_tensor.repeat = partial(paddorch.Tensor.repeat, paddle_tensor)
#     paddle_tensor.add = partial(paddorch.Tensor.add, paddle_tensor)
#     paddle_tensor.item = partial(paddorch.Tensor.item, paddle_tensor)
#     paddle_tensor.t = partial(paddorch.Tensor.t, paddle_tensor)
#     paddle_tensor.reshape = partial(paddorch.Tensor.reshape, paddle_tensor)
#     # paddle_tensor.shape=partial(paddorch.Tensor.shape,paddle_tensor)
#     # paddle_tensor.__setitem__=partial(paddorch.Tensor.__setitem__,paddle_tensor)
#     # paddle_tensor.__getitem__=partial(paddorch.Tensor.__getitem__,paddle_tensor)
#     paddle_tensor.index_copy_ = partial(paddorch.Tensor.index_copy_, paddle_tensor)
#     paddle_tensor.index_copy = partial(paddorch.Tensor.index_copy, paddle_tensor)
#     paddle_tensor.new_empty = partial(paddorch.Tensor.new_empty, paddle_tensor)
#     paddle_tensor.view_as = partial(paddorch.Tensor.view_as, paddle_tensor)
#     paddle_tensor.clamp = partial(paddorch.Tensor.clamp, paddle_tensor)
#     paddle_tensor.requires_grad_ = partial(paddorch.Tensor.requires_grad_, paddle_tensor)
#     paddle_tensor.set_gradient = partial(paddorch.Tensor.set_gradient, paddle_tensor)
#     paddle_tensor.backward = partial(paddorch.Tensor.backward, paddle_tensor)
#     paddle_tensor.new_zeros = partial(paddorch.Tensor.new_zeros, paddle_tensor)
#     paddle_tensor.new_ones = partial(paddorch.Tensor.new_ones, paddle_tensor)
#     paddle_tensor.sort = partial(paddorch.Tensor.sort, paddle_tensor)
#     paddle_tensor.index_select = partial(paddorch.Tensor.index_select, paddle_tensor)
#     paddle_tensor.masked_fill_ = partial(paddorch.Tensor.masked_fill_, paddle_tensor)
#     paddle_tensor.masked_fill = partial(paddorch.Tensor.masked_fill, paddle_tensor)
#     paddle_tensor.argmax = partial(paddorch.Tensor.argmax, paddle_tensor)
#     paddle_tensor.tolist = partial(paddorch.Tensor.tolist, paddle_tensor)
#     paddle_tensor.uniform_ = partial(paddorch.Tensor.uniform_, paddle_tensor)
#     paddle_tensor.__getstate__ = partial(paddorch.Tensor.__getstate__, paddle_tensor)
#     paddle_tensor.__setstate__ = partial(paddorch.Tensor.__setstate__, paddle_tensor)
#     paddle_tensor.byte = partial(paddorch.Tensor.byte, paddle_tensor)
#     paddle_tensor.bernoulli_ = partial(paddorch.Tensor.bernoulli_, paddle_tensor)
#     paddle_tensor.bool = partial(paddorch.Tensor.bool, paddle_tensor)
#     paddle_tensor.chunk = partial(paddorch.Tensor.chunk, paddle_tensor)
#     paddle_tensor.__invert__ = partial(paddorch.Tensor.__invert__, paddle_tensor)
#     paddle_tensor.split = partial(paddorch.Tensor.split, paddle_tensor)
#     paddle_tensor.device = ""
#     return paddle_tensor