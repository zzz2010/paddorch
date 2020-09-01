import paddle.fluid as fluid
import paddle
from paddle.fluid.layer_helper import LayerHelper

import paddle_torch.cuda
import paddle_torch.nn
import os
import paddle_torch.nn.functional
import paddle_torch.nn.init
import paddle_torch as torch
from paddle.fluid import dygraph
import numpy as np

def varbase_to_tensor(x):
    return Tensor(x)

    y2 =  new_full( x.shape, 0)
    y=fluid.layers.assign(x)
    y2.block=y.block
    # fluid.layers.assign(x.gradient(), y.gradient())
    return y2

def new_full(size, fill_value, dtype=None,  requires_grad=False):
    if dtype is None:
        dtype='float32'
    x=Tensor(np.full(size,fill_value,dtype=dtype))
    x.stop_gradient=not requires_grad
    return x


class Tensor(dygraph.core.VarBase):
    def __init__(self,*args, **kwargs):

        if isinstance(args[0],dygraph.core.VarBase):

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

    def sum(self,dim=None, keep_dim=False):
        return torch.sum(self,dim=dim,keep_dim=keep_dim)

    def mean(self,dim=None, keep_dim=False):
        return torch.mean(self,dim=dim,keep_dim=keep_dim)

    def max(self,dim=None, keep_dim=False):
        return torch.max(self,dim=dim,keep_dim=keep_dim)

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

    def pow(self,k):
        return torch.pow(self,k)

    def dim(self) :
        return len(self.shape)

    def clone(self):
        y = self.new_full(self.shape, 0)
        fluid.layers.assign(self, y)
        return y

    def clamp_(self,min,max):
        self.set_value( fluid.layers.clip(self,float(min),float(max) ) )
        return self

    def float(self):
        return self.astype('float32')

    def add_(self,x):
        self.set_value(x+self)
        return self

    def expand(self,*sizes):
        ##handle -1 case
        expand_times=[ x//y if x>=y else 1 for x,y in zip(sizes,self.shape) ]
        x= varbase_to_tensor(paddle.fluid.layers.expand(self, expand_times, name=None))
        return x

    def div_(self,x):
        self.set_value(self/x)
        return  self

    def copy_(self,src):
        fluid.layers.assign(src,self)
        return self

    def mul(self,x):
        return varbase_to_tensor(self*x)

    def permute(self,*perm):
        x= fluid.layers.transpose(self, perm, name=None)
        return varbase_to_tensor(x)

    def transpose(self,*perm):
        return self.permute(perm)

    def cpu(self,*args, **kwargs):
        return self

    def cuda(self,*args, **kwargs):
        return self

    def to(self,*args, **kwargs):
        return self
    def contiguous(self):
        return self


    def view(self,*size):
        x= fluid.layers.reshape(self,size)

        return varbase_to_tensor(x)

    def repeat(self,*size):
        x=np.tile(self.numpy(),size )
        return Tensor(x)




    def item(self):
        return self.numpy().flatten()[0]

