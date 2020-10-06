import paddle.fluid as fluid
import paddle
from paddle.fluid.layer_helper import LayerHelper

import paddorch.cuda
import paddorch.nn
import os
import paddorch.nn.functional
import paddorch.nn.init
import paddorch as torch
from paddle.fluid import dygraph
import numpy as np

def varbase_to_tensor(x):
    return Tensor(x)


def new_full(size, fill_value, dtype=None,  requires_grad=False):
    if dtype is None:
        dtype='float32'
    x=Tensor(np.full(size,fill_value,dtype=dtype))
    x.stop_gradient=not requires_grad
    return x


class Tensor(dygraph.core.VarBase):
    def __init__(self,*args, **kwargs):

        if isinstance(args[0],dygraph.core.VarBase) or isinstance(args[0],dygraph.core.LoDTensor):

            super(Tensor, self).__init__( args[0].dtype,args[0].shape,args[0].name,dygraph.core.VarDesc.VarType.LOD_TENSOR, True)

            fluid.layers.assign(args[0],self)
        else:
            super(Tensor, self).__init__(*args, **kwargs)
            # self=self #dygraph.core.VarBase(*args, **kwargs)

        self.device=None
        # self.block=self.block
        # self.dtype=self.dtype
        # self.name=self.name
        # self.persistable=self.persistable
        # self.shape=self.shape
        # self.stop_gradient=self.stop_gradient
        # self.type=self.type

    def dim(self):
        return len(self.shape)

    def min(self):
        return torch.min(self)

    def sum(self,dim=None, keepdim=False):
        return torch.sum(self,dim=dim,keepdim=keepdim)

    def mean(self,dim=None, keepdim=False):
        return torch.mean(self,dim=dim,keepdim=keepdim)

    def max(self,dim=None, keepdim=False):
        return torch.max(self,dim=dim,keepdim=keepdim)

    def new_full(self, size, fill_value, dtype=None, device=None, requires_grad=False):
        return new_full( size, fill_value, dtype,requires_grad)

    def _fill_(self, val):
        fluid.layers.fill_constant(self.shape,self.dtype,val,out=self)  #assign(self.new_full(self.shape,val),self)
        return self

    def fill_(self, val,dim=None,indices=None):
        if dim==None:
            return self._fill_(val)
        if len(indices)==0:
            return self
        if dim==0:
            x_numpy = self.numpy()
            x_numpy[indices] =val
            self.set_value(x_numpy)
        if dim==1:
            x_numpy = self.numpy()
            x_numpy[:,indices] = val
            self.set_value(x_numpy)
        if dim==2:
            x_numpy = self.numpy()
            x_numpy[:,:,indices] = val
            self.set_value(x_numpy)
        return self

    def size(self,dim=None):
        if dim is None:
            return self.shape
        return self.shape[dim]

    def unsqueeze(self, dim ):
        x=fluid.layers.unsqueeze(self, dim)
        return varbase_to_tensor(x)

    def narrow(self, dim, start, length):
        return varbase_to_tensor(fluid.layers.slice(self,[dim],[start],[start+length] ))

    def squeeze(self, dim=[] ):
        if isinstance(dim,int):
            dim=[dim]
        x= fluid.layers.squeeze(self, dim)
        return varbase_to_tensor(x)

    def bmm(self,x):
        return Tensor(fluid.layers.bmm(self,x))
    def sqrt(self):
        return torch.sqrt(self)
    def normal_(self,m,std):
        fluid.layers.assign(fluid.layers.randn(self.shape)*std+m,self)
        return self

    def random_(self,low,high):

        fluid.layers.assign(fluid.layers.randint(low,high,self.shape) ,self)
        return self

    def pow(self,k):
        return torch.pow(self,k)

    def dim(self) :
        return len(self.shape)

    def clone(self):
        y = self.new_full(self.shape, 0,dtype=str(self.dtype).replace("VarType.","").lower().replace("fp32","float32") )
        fluid.layers.assign(self, y)
        return y

    def clamp_(self,min,max):
        self.set_value( fluid.layers.clip(self,float(min),float(max) ) )
        return self

    def float(self):
        return self.astype('float32')

    def dot(self,x):
        return torch.dot(self,x)
    def add_(self,x):
        self.set_value(x+self)
        return self
    def matmul(self,y):
        return torch.matmul(self,y)

    def norm(self,dim=-1, keepdim=True):
        return torch.norm(self,dim=dim,keepdim=keepdim)

    def expand(self,*sizes):
        ##handle -1 case
        expand_times=[ x//y if x>=y else 1 for x,y in zip(sizes,self.shape) ]
        x= varbase_to_tensor(paddle.fluid.layers.expand(self, expand_times, name=None))
        return x

    def div_(self,x):
        self.set_value(self/x)
        return  self

    def copy_(self,src):
        torch.copy(src,self)
        return self

    def mm(self,x):
        return torch.mm(self,x)
    def mul(self,x):
        return varbase_to_tensor(self*x)

    def permute(self,*perm):
        x= fluid.layers.transpose(self, perm, name=None)
        return varbase_to_tensor(x)

    def transpose(self,*perm):
        if len(perm)==2 and len(self.shape)>2:
            ###only swap two axis
            perm2=list(range(len(self.shape)))
            a=perm2[perm[0]]
            perm2[perm[0]]=perm[1]
            perm2[perm[1]] =a
            perm=perm2
        return self.permute(*perm)

    def cpu(self,*args, **kwargs):
        return self

    def cuda(self,*args, **kwargs):
        return self

    def to(self,*args, **kwargs):
        return self
    def contiguous(self):
        return self

    def flip(self,dim):
        return torch.flip(self,dim)
    def view(self,*size):
        x= fluid.layers.reshape(self,size)

        return varbase_to_tensor(x)

    def repeat(self,*size):
        x=np.tile(self.numpy(),size )
        return Tensor(x)


    def add(self,x):
        return Tensor(self+x)

    def __add__(self, other):
        return Tensor(super(Tensor, self).__add__(other) )
    def __sub__(self, other):
        return Tensor( super(Tensor, self).__sub__(other))

    def __div__(self, other_var):
        return Tensor(super(Tensor, self).__div__(other_var))

    def __mul__(self, other_var):
        return Tensor(super(Tensor, self).__mul__(other_var))

    def item(self):
        return self.numpy().flatten()[0]

    def t(self):
        return Tensor(fluid.layers.transpose(self,np.arange(len(self.shape))[::-1]))
    def reshape(self,*size):
        if len(size)==1:
            size=size[0]
        return self.view(*size)

    def __getitem__(self,args):
        from typing import   Iterable
        if not isinstance(args,Iterable):
            return Tensor(super(Tensor, self).__getitem__(args))
        if isinstance(args[0],dygraph.core.VarBase):
            if isinstance(args,tuple):
                if len(args)==2:
                    return torch.take(self, list(zip(args[0].numpy().astype(int).tolist(), args[1].numpy().astype(int).tolist())))
                else:
                    raise("not support more than 2 axis array indexing")
            else:
                return torch.take(self,
                                  args[0].numpy().astype(int).tolist())

        return Tensor(super(Tensor, self).__getitem__(args))