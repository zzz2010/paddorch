import paddle
tensor=paddle.randn((3,4))
dlpack=tensor.value().get_tensor()._to_dlpack()
tensor_from_dlpack = paddle.fluid.core.from_dlpack(dlpack)
tensor_from_dlpack.__class__=paddle.fluid.LoDTensor

bb=paddle.Tensor( tensor_from_dlpack )
bb=bb.cpu()
print(bb)
paddle.set_device()
tensor_from_dlpack.__class__=paddle.fluid.core_avx.Tensor
# paddle.fluid.dygraph.to_variable( tensor_from_dlpack)
