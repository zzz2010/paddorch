import paddle.fluid as fluid
import paddle
import paddle_torch.cuda
import paddle_torch.nn
import os
import paddle_torch.nn.functional
import paddle_torch.nn.init
from paddle.fluid import dygraph
import numpy as np
from paddle_torch.tensor import varbase_to_tensor,Tensor
import paddle_torch.optim

double="float32"

def chunk(self , chunks , dim ):
    slices= fluid.layers.unstack(self, axis=dim, num=None)
    out_list=[]
    step=int(np.ceil(len(slices)/chunks))
    for st in range(0,len(slices),step):
        out_list.append(varbase_to_tensor(fluid.layers.concat( [paddle.fluid.layers.unsqueeze(x, dim, name=None) for x in slices[st:(st+step)] ], axis=dim, name=None)))
    return out_list

def tensor(x,dtype=np.float32):
    if isinstance(x,list):
        x=np.array(x,dtype=dtype)
    return Tensor(x)

def FloatTensor(x):
    return tensor(x)

def abs(x):
    return fluid.layers.abs(x)
def max(x,dim=None,keepdim=False):
    return varbase_to_tensor(fluid.layers.reduce_max(x,dim,keep_dim= keepdim ))

def min(x,dim=None,keepdim=False):
    return varbase_to_tensor(fluid.layers.reduce_min(x,dim,keep_dim= keepdim ))

def full_like(x,fill_value):
    return Tensor.new_full(x,x.shape,fill_value)



def where(condition, x=None, y=None):
    if  x is None:
        return fluid.layers.where(condition)
    else:
        w=condition.astype("float32")
        z=w*x+(1-w)*y
        return z
        # for i, flag in  enumerate(condition.numpy()) :
        #     if flag:
        #         out.append(x[i])
        #     else:
        #         out.append(y[i])
        # return stack(out)

def take(x,indices):
    if len(indices[0])==2:
        return stack([x[a[0],a[1]] for a in indices])
    if len(indices[0]) == 1:
        return stack([x[a[0]] for a in indices])

def copy(src,target):
    target.set_value(src)
    # fluid.layers.assign(src,target)
def rsqrt(x):
    return varbase_to_tensor(paddle.fluid.layers.rsqrt(x, name=None))

def sum(x,dim=None, keepdim=False):
    return varbase_to_tensor(fluid.layers.reduce_sum(x,dim,keepdim))

def sqrt(x):
    return varbase_to_tensor(fluid.layers.sqrt(x))
def pow(x,y):
    return varbase_to_tensor(fluid.layers.pow(x,y))

def as_tensor(x,dtype=np.float32):
    return tensor(x,dtype)

def is_tensor(x):
    return isinstance(x, ( dygraph.core.VarBase,  dygraph.framework.Variable,
                        dygraph.framework.ComplexVariable))

def manual_seed(seed):
    fluid.Program.random_seed=seed
    np.random.seed(seed)

def multinomial(weights , num_samples , replacement=False):
    select_samples=[]
    if not replacement and num_samples>len(weights):
        raise(Exception("num_samples should be no greater than the weights length"))

    while len(select_samples)<num_samples:
        x=fluid.layers.sampling_id(fluid.layers.reshape(weights,[1,-1] ))
        if not replacement:
            if x in select_samples:
                continue
        select_samples.append(x)
    return  select_samples
def LongTensor(x):
    if isinstance(x,int):
        return Tensor(fluid.Tensor)
    if isinstance(x,list):
        x=np.array(x,dtype=np.int32)
    return Tensor(x )

def stack(inputs,dim=0,out=None):
    x= fluid.layers.stack(inputs ,axis=dim )
    if out is None:
        return varbase_to_tensor(x)
    else:
        fluid.layers.assign(x,out)
        return out
def arange(*args,**kwargs):
    return Tensor(np.arange(*args,**kwargs).astype("int32"))
    # if end==0:
    #     return []
    # return varbase_to_tensor(paddle.fluid.layers.range(0, end, step, dtype))

def device(name):
    if name.startswith("cuda"):
        device_id=name.replace("cuda","").replace(":","")
        if len(device_id)==0:
            return fluid.CUDAPlace(0)
        else:
            return fluid.CUDAPlace(int(device_id))
    else:
        return fluid.CPUPlace()

def cat(tensors, dim=0, out=None):
    x=fluid.layers.concat(tensors,axis=dim)
    if out is None:
        return varbase_to_tensor(x)
    else:
        fluid.layers.assign(x,out)
        return out


def ones(*size, out=None, dtype="float32",device=None):
    return varbase_to_tensor(fluid.layers.ones(size,dtype))

def zeros(*size, out=None, dtype="float32",device=None):
    return varbase_to_tensor(fluid.layers.zeros(size,dtype))

def ones_like(x, out=None,device=None):
    return varbase_to_tensor(fluid.layers.ones_like(x,out))


def zeros_like(x, out=None,device=None):
    return varbase_to_tensor(fluid.layers.zeros_like(x,out))

def randn(*shape):
    return varbase_to_tensor(fluid.layers.randn(shape))

def mean(input):
    return  fluid.layers.mean(input)

def var(input, dim=None, keepdim=False, unbiased=True, name=None):
    if isinstance(dim,tuple):
        dim=list(dim)
    rank = len(input.shape)
    dims = dim if dim != None and dim != [] else range(rank)
    dims = [e if e >= 0 else e + rank for e in dims]
    inp_shape = input.shape
    mean = fluid.layers.reduce_mean(input, dim=dim, keep_dim=True, name=name)
    tmp = fluid.layers.reduce_mean((input - mean)**2, dim=dim, keep_dim=keepdim, name=name)
    if unbiased:
        n = 1
        for i in dims:
            n *= inp_shape[i]
        factor = n / (n - 1.0) if n > 1.0 else 0.0
        tmp *= factor
    return tmp
def mean(input, dim=None, keepdim=False, out=None):
    if isinstance(dim,tuple):
        dim=list(dim)
    x= fluid.layers.reduce_mean(input,dim,keepdim)
    if out is None:
        return varbase_to_tensor(x)
    else:
        fluid.layers.assign(x,out)
        return out

def lerp(input, end, weight, out=None):
    x= input+float(weight)*(end-input)
    if out is None:
        return x
    else:
        fluid.layers.assign(x,out)
        return out

def flatten(x,dim=1):
    x=fluid.layers.flatten(x ,axis=dim)
    return varbase_to_tensor(x)


def clamp(input, min, max, out=None) :
    return varbase_to_tensor(fluid.layers.clip(input, min, max))



def  no_grad(func=None):
    return fluid.dygraph.no_grad(func)

def check_if_state_dict(obj):
    '''
        check if the obj is a state_dict from paddle model
    '''
    first_key=list(obj.keys())[0]
    try:
        obj[first_key].name
        return True
    except:
        return False
def save(dict_obj, filename):
    '''
    save dict of state dict as a folder , or state dict as a file
    '''
    if check_if_state_dict(dict_obj):
        if filename.endswith(".pdparams"):
            filename=filename.replace(".pdparams","")
        fluid.dygraph.save_dygraph(dict_obj,filename)
    else:
        os.makedirs(filename,exist_ok=True)
        for key in dict_obj:
            fluid.dygraph.save_dygraph( dict_obj[key], filename+"/"+str(key) )



def load(file_path,map_location=None) :
    import glob
    out_dict=dict()
    if os.path.isdir(file_path):
        for fn in glob.glob(file_path+"/*.pdparams"):
            key=os.path.basename(fn).replace(".pdparams","")
            out_dict[key]=fluid.dygraph.load_dygraph(fn)[0]

    else:
        out_dict=fluid.dygraph.load_dygraph(file_path)[0]

    return out_dict


