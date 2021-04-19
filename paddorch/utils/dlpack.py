from paddle import  fluid
import numpy as np

def to_lodtensor(data, place):
  # 存储Tensor的长度作为LoD信息
  seq_lens = [len(seq) for seq in data]
  cur_len = 0
  lod = [cur_len]
  for l in seq_lens:
      cur_len += l
      lod.append(cur_len)
  # 对待转换的 Tensor 降维
  flattened_data = np.concatenate(data, axis=0).astype("int64")
  flattened_data = flattened_data.reshape([len(flattened_data), 1])
  # 为 Tensor 数据添加lod信息
  res = fluid.LoDTensor()
  res.set(flattened_data, place)
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
    tensor = to_lodtensor(tensor,fluid.CPUPlace())
    dltensor = tensor._to_dlpack()
    return dltensor

def from_dlpack(dlpack):
    tensor_from_dlpack = fluid.core.from_dlpack(dlpack)
    return tensor_from_dlpack


