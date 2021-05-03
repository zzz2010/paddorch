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
from typing import   Iterable
def varbase_to_tensor(x):
    return convertTensor(x)


def new_full(size, fill_value, dtype=None,  requires_grad=False):
    if dtype is None:
        dtype='float32'

    x=convertTensor(paddle.full(size,fill_value,dtype=dtype))
    x.stop_gradient=not requires_grad
    return x

def convertTensor(x):
    if isinstance(x,paddorch.Tensor):
        return x
    return  paddorch.Tensor(x)

# class Tensor(dygraph.core.VarBase):
class Tensor(paddle.Tensor):
    def __init__(self,*args, **kwargs):
        if isinstance(args[0],dygraph.core.VarBase) or isinstance(args[0],dygraph.core.LoDTensor):

            super(Tensor, self).__init__( args[0].dtype,args[0].shape,args[0].name,dygraph.core.VarDesc.VarType.LOD_TENSOR, True)

            fluid.layers.assign(args[0],self)
        elif isinstance(args[0],Iterable):
            args=list(args)
            if isinstance(args[0][0],int):
                args[0] = np.array(args[0]).astype("int64")
            else:
                args[0]=np.array(args[0]).astype("float32")
            super(Tensor, self).__init__(*args, **kwargs)
        elif isinstance(args[0],int):
            super(Tensor, self).__init__(np.zeros(args).astype("float32") )
        else:
            super(Tensor, self).__init__(*args, **kwargs)
            # self=self #dygraph.core.VarBase(*args, **kwargs)

        self.device=str(self.place)
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
        if dtype is not None:
            dtype=dtype.replace("fp","float")
        else:
            dtype=self.dtype
        return new_full( size, fill_value, dtype,requires_grad)

    def new(self,*size):
        return new_full(size,0)

    def scatter_add_(self, dim,index, updates ):
        assert  dim==0, "scatter_add_, no support dim>0"
        if len(index.shape)==1:
            paddle.scatter_(self, index , updates.astype("float32"), overwrite=False)
        else:
            for ii in range(index.shape[1]):
                paddle.scatter_(self,index[:,ii],updates.astype("float32"),overwrite=False)

        return self

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
        y = self.new_full(self.shape, 0,dtype=str(self.dtype).replace("VarType.","").lower().replace("paddle.","").replace("fp32","float32") )
        fluid.layers.assign(self, y)
        y.stop_gradient=self.stop_gradient
        return y

    def clamp_(self,min,max):
        self.set_value( fluid.layers.clip(self,float(min),float(max) ) )
        return self

    def float(self):
        return convertTensor(self.astype('float32'))
    def long(self):
        return convertTensor(self.astype('int64'))

    def dot(self,x):
        return torch.dot(self,x)
    def add_(self,x):
        self.set_value(x+self)
        return self
    def matmul(self,y):
        return torch.matmul(self,y)

    def norm(self,p=2,dim=-1, keepdim=True):
        return torch.norm(self,p=p,dim=dim,keepdim=keepdim)

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

    def mul_(self,x):
        y= self.mul(x)
        self.copy_(y)
        return  self

    def add_(self,x):
        y= self.add(x)
        self.copy_(y)
        return  self

    def permute(self,*perm):
        x=paddle.transpose(self,perm)

        return varbase_to_tensor(x)

    def transpose(self,*perm):
        # if len(perm)==2 and len(self.shape)>2:
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


    def type(self,dtype):
        return self.astype(dtype)

    def contiguous(self):
        return self

    def flip(self,dim):
        return torch.flip(self,dim)
    def view(self,*size):
        if len(size)==1:
            if isinstance(size[0],Iterable):
                size=size[0]
        x= paddle.reshape(self,size)

        return varbase_to_tensor(x)

    def repeat(self,*size):
        return paddorch.repeat(self,*size)
        # x=paddle.tile(self,size)
        # return convertTensor(x)


    def add(self,x):
        return convertTensor(self+x)

    def __mod__(self, other):
        return convertTensor(super(Tensor, self). __mod__(other))
    def __add__(self, other):
        return convertTensor(super(Tensor, self).__add__(other) )
    def __sub__(self, other):
        return convertTensor( super(Tensor, self).__sub__(other))

    def __div__(self, other_var):
        return convertTensor(super(Tensor, self).__div__(other_var))

    def __truediv__(self, other_var):
        return self.__div__( other_var)

    def __mul__(self, other_var):
        return convertTensor(super(Tensor, self).__mul__(other_var))

    def item(self):
        return self.numpy().flatten()[0]

    def t(self):
        return convertTensor(paddle.transpose(self,paddle.arange(len(self.shape))[::-1]))
        # return convertTensor(fluid.layers.transpose(self,np.arange(len(self.shape))[::-1]))
    def reshape(self,*size):
        if len(size)==1:
            size=size[0]
        return self.view(*size)

    def __setitem__(self, key, value):
        if isinstance(key,tuple):
            if len(key)==2:
                return super(Tensor,self).__setitem__(key,value)
        if   isinstance(key,int):
            return super(Tensor, self).__setitem__(key, value)

        def convert_key_to_inttensor(key):
            if isinstance(key, np.ndarray):
                # print(max(args),min(args),self.shape,len(args),len(set(args)) )
                key=paddorch.from_numpy(key).long()
                # print("converted numpy", type(args))
            if  isinstance(key,paddle.Tensor):
                if key.dtype==paddle.fluid.core.VarDesc.VarType.BOOL:
                    key = paddle.masked_select(paddle.arange(len(key)), key)
                elif key.dtype==paddle.fluid.core.VarDesc.VarType.INT32 or key.dtype==paddle.fluid.core.VarDesc.VarType.INT64:
                    return key
                else:
                    return key.astype("int64")
            if isinstance(key,int):
                return paddorch.LongTensor(np.array([key]))
            if isinstance(key,list):
                key = paddorch.from_numpy(key).long()
            return key

        if isinstance(key, np.ndarray) or isinstance(key,paddle.Tensor):
            key = convert_key_to_inttensor(key)
        elif isinstance(key,Iterable) :
            key2=[]
            for i in range(len(key)):
                key2.append(convert_key_to_inttensor(key[i]))

            key=paddle.stack(key2,axis=1)

            if len(key2)==1:
                key= key.reshape([-1])

        else:
            key=convert_key_to_inttensor(key)
        return paddle.scatter_(self,key,value)


    def __getitem__(self,args):
        from typing import   Iterable

        if isinstance(args, np.ndarray):
            # print(max(args),min(args),self.shape,len(args),len(set(args)) )
            args=paddorch.from_numpy(args).long()
            # print("converted numpy", type(args))
        if  isinstance(args,paddle.Tensor):
            if args.dtype==paddle.fluid.core.VarDesc.VarType.BOOL:
                return convertTensor(paddle.masked_select(self,args))
            elif args.dtype==paddle.fluid.core.VarDesc.VarType.INT32 or args.dtype==paddle.fluid.core.VarDesc.VarType.INT64:
                return convertTensor(paddle.index_select(self,args,axis=0))
        if isinstance(args,Iterable) and (len(args)==2):
            if isinstance(args[1],paddle.Tensor) and (args[1].dtype==paddle.fluid.core.VarDesc.VarType.BOOL):
                sel_indices=paddle.masked_select(paddle.arange(len(args[1])),args[1])  #paddle.arange(len(args[1]))[args[1]]
                return convertTensor(paddle.index_select(self[args[0]],sel_indices,axis=1))
            elif isinstance(args[0],paddle.Tensor):
                    return convertTensor(super(Tensor, self).__getitem__(args[0])[args[1]])


        return convertTensor(super(Tensor, self).__getitem__(args))



    def index_copy_(self,dim, index, tensor):
        return paddorch.index_copy_(self,dim, index, tensor)

    def index_copy(self,dim, index, tensor):
        return paddorch.index_copy(self,dim, index, tensor)

    def new_empty(self,size, dtype=None, device=None, requires_grad=False):
        return paddorch.empty(size).astype(dtype)

    def view_as(self,Y):
        return self.view(*Y.shape)

    def clamp(self,*args,**kwargs):
        return paddorch.clamp(self,*args,**kwargs)

    def requires_grad_(self):
        self.stop_gradient=False
        return self

    def set_gradient(self, gradient=None):
        def set_grad(grad):
            return gradient
        if gradient is not None:
            try:##only work in the dev version
                helper=self.register_hook(set_grad)
            except:
                pass

    def  backward(self, gradient=None, retain_graph=False):
        def set_grad(grad):
            print("set_grad",gradient)
            grad.set_value(gradient)
            return grad
        if gradient is not None:
            try:##only work in the dev version
                helper=self.register_hook(set_grad)
            except:
                pass

            ret = super(Tensor, self).backward(retain_graph=retain_graph)

        if gradient is not None:
            try:  ##only work in the dev version
                helper.remove()
            except:
                pass
        return

    @property
    def shape(self):
        shape=super(Tensor, self).shape
        if isinstance(shape,int):
            return tuple(shape)
        if isinstance(shape[0],Iterable):
            shape=shape[0]
        return tuple( shape)


    @property
    def grad(self):
        if super(Tensor, self).grad is None:
            return None
        return convertTensor(super(Tensor, self).grad)

    def detach(self):
        return convertTensor(super(Tensor, self).detach() )

    def new_zeros(self,*size):
        return paddorch.zeros(*size,dtype=self.dtype)
    def new_ones(self,*size):
        return paddorch.ones(*size,dtype=self.dtype)


    def sort(self,dim=-1, descending=False):
        order= paddorch.argsort(self[:,dim], descending=descending)
        return  self[order],order


    def index_select(self,dim, index):
        index=index.astype("int64")

        ret= convertTensor(paddle.index_select(self,index=index, axis=dim))
        return  ret

    def masked_fill_(self, mask,value):
        mask=paddle.expand_as(mask,self)
        new_values=paddle.where(mask,self,paddle.ones(self.shape)*value)
        fluid.layers.assign(new_values, self)
        return self

