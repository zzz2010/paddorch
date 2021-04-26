import numpy as np
import paddle
import paddorch
def double_hook(grad):
    print("double_hook1",grad,grad*6)
    return grad*6

def double_hook2(grad):
    print("double_hook2",grad,grad * 2)
    return grad * 2
x = paddorch.Tensor(paddle.to_tensor([0., 1., 2., 3.]))
y = paddorch.Tensor(paddle.to_tensor([4., 5., 6., 7.]))
x.stop_gradient = False
y.stop_gradient = False

w = x + y
w.stop_gradient = False
helper = w.register_hook(double_hook)
helper = w.register_hook(double_hook2)

z = paddorch.Tensor(paddle.to_tensor([1., 2., 3., 4.]))
z.stop_gradient = False

o = z.matmul(w)

# # remove hook before backward
# helper.remove()

o.backward(paddorch.Tensor(np.array([10. ])))
print("x.grad:",x.grad)
print("o.grad:",o.grad)
print("w.grad:",w.grad)
print("y.grad:",y.grad)