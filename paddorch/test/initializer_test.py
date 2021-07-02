from paddle import fluid
import paddorch as torch
import paddle
from scipy.sparse import csr_matrix
from paddorch.nn.init import xavier_uniform_
from paddorch.sparse import  FloatTensor
import numpy as np
place = fluid.CPUPlace()
with fluid.dygraph.guard(place=place):
    i = torch.from_numpy(np.array([[0, 2], [1, 0], [1, 2]]) ).astype("int32")
    v = paddle.Tensor(np.array([3, 4, 5])).astype("float32")
    x=FloatTensor(i,v,(2,3))
    print(x)

    A = csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]]).todense().astype("float32")
    x= paddle.Tensor(A)
    # x = torch.randn((4, 23, 16))
    print(xavier_uniform_(x))



