import paddorch

arr=paddorch.convertTensor( 1.0/paddorch.arange(1,30))
print(arr)

arr[arr>0.1]=0.0
print(arr)

arr[paddorch.isfinite(arr)]=1.0
print(arr)


arr[paddorch.isinf(arr)]=2.0
print(arr)


arr=paddorch.convertTensor( 1.0/paddorch.arange(1,31)).view(-1,5,2)
print("before",arr)
arr[:,2,:]=999
print("after",arr)