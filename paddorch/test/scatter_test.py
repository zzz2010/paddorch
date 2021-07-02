import paddle

a=paddle.zeros([1]).astype("float32")
b=paddle.zeros([5]).astype("int32")
c=paddle.arange(5,10).astype("float32")

d=paddle.scatter(a,b,c,overwrite=False)
print(d)