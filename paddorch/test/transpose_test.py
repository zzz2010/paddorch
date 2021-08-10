import paddle
import paddorch.monkeypatch
x = paddle.randn([2, 3, 4])
x_transposed = paddle.transpose(x, perm=[1, 0, 2])
print(x_transposed.shape)


x_transposed=x.transpose([1, 0, 2])
print(x_transposed.shape)