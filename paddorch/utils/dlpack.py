from paddle import  fluid
import numpy as np
import paddorch
import paddle

def to_lodtensor(data, place):
  # 存储Tensor的长度作为LoD信息
  seq_lens = [len(seq) for seq in data]
  cur_len = 0
  lod = [cur_len]
  for l in seq_lens:
      cur_len += l
      lod.append(cur_len)
  # 对待转换的 Tensor 降维
  # if data.shape[0]>0:
  #   flattened_data = paddorch.cat(data, dim=0).astype("int64")
  # else:
  flattened_data=paddorch.convertTensor(data)
  flattened_data = flattened_data.view(-1)
  # 为 Tensor 数据添加lod信息
  res = fluid.LoDTensor( )
  res.set(flattened_data.numpy(), place,zero_copy=False)
  res.set_lod([lod])
  return res

def LodTensor_to_Tensor(lod_tensor):
  # 获取 LoD-Tensor 的 lod 信息
  lod = lod_tensor.lod()
  # 转换成 array
  array = np.array(lod_tensor)
  new_array = []
  # 依照原LoD-Tensor的层级信息，转换成Tensor
  for i in range(len(lod[0]) - 1):
      new_array.append(array[lod[0][i]:lod[0][i + 1]])
  return new_array

def to_dlpack(tensor):
    # tensor = to_lodtensor(tensor,fluid.CPUPlace())
    # p = fluid.core.Place()
    # p.set_place(paddle.CPUPlace())
    # dltensor = tensor._copy(p)._to_dlpack()
    import nnabla as nn ##pip install nnabla==1.18.0
    from nnabla.utils.dlpack import to_dlpack
    from nnabla.ext_utils import get_extension_context
    ctx = get_extension_context('cpu')
    nn.set_default_context(ctx)
    a = nn.NdArray.from_numpy_array(tensor.numpy() )
    return to_dlpack(a)


def from_dlpack(dlpack):
    tensor_from_dlpack = fluid.core.from_dlpack(dlpack)
    if "int64" in str(tensor_from_dlpack):
        return paddorch.convertTensor(paddorch.Tensor(np.array(tensor_from_dlpack)).astype("int64"))
    else:
        return paddorch.Tensor(np.array(tensor_from_dlpack))


