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

enable_monkeypatch=True

def varbase_to_tensor(x):
    return convertTensor(x)



def new_full(size, fill_value, dtype=None,  requires_grad=False):
    if dtype is None:
        dtype='float32'

    x=convertTensor(paddle.full(size,fill_value,dtype=dtype))
    x.stop_gradient=not requires_grad
    return x

def convertTensor(x):
    if enable_monkeypatch:
        if isinstance(x,paddle.Tensor):
            return  x
    if isinstance(x,paddorch.Tensor):
        return x
    ret=  paddorch.Tensor(x)

    return ret

# class Tensor(dygraph.core.VarBase):
class Tensor(paddle.Tensor  ):
    def __init__(self,*args, **kwargs):

        if isinstance(args[0],dygraph.core.VarBase) or isinstance(args[0],dygraph.core.LoDTensor):
            dtype=args[0].dtype
            super(Tensor, self).__init__( dtype,args[0].shape,args[0].name,dygraph.core.VarDesc.VarType.LOD_TENSOR, True)

            fluid.layers.assign(args[0],self)
        elif isinstance(args[0],Iterable):
            args=list(args)
            if isinstance(args[0][0],int):
                args[0] = np.array(args[0]).astype("int32")
            else:
                args[0]=np.array(args[0]).astype("float32")
            super(Tensor, self).__init__(*args, **kwargs)
        elif isinstance(args[0],int):
            super(Tensor, self).__init__(np.zeros(args).astype("float32") )
        else:
            super(Tensor, self).__init__(*args, **kwargs)
            # self=self #dygraph.core.VarBase(*args, **kwargs)

        # self.block=self.block
        # self.dtype=self.dtype
        # self.name=self.name
        # self.persistable=self.persistable
        # self.shape=self.shape
        # self.stop_gradient=self.stop_gradient
        # self.type=self.type

    @property
    def device(self):
        return str(self.place)

    @property
    def is_cuda(self):
        if "cuda" in str(self.place):
            return True
        else:
            return  False

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
        if "64" in str(updates.dtype):
            updates=updates.astype("float32")

        if "64" in str(self.dtype):
            self=self.astype("float32")
        if len(index.shape)==1:
            paddle.scatter_(self, index , updates.astype("float32"), overwrite=False)
        else:
            for ii in range(index.shape[1]):
                paddle.scatter_(self,index[:,ii],updates.astype("float32"),overwrite=False)

        return self

    def scatter_add(self, dim,index, updates ):
        assert  dim==0, "scatter_add_, no support dim>0"
        if "64" in str(updates.dtype):
            updates=updates.astype("float32")
        ret=self
        if "64" in str(ret.dtype):
            ret=ret.astype("float32")
        if len(index.shape)==1:
            ret=paddle.scatter(ret, index , updates , overwrite=False)
        else:
            for ii in range(index.shape[1]):
                ret=paddle.scatter(ret,index[:,ii],updates ,overwrite=False)

        return ret

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
        paddorch.copy(self, y)
        y.stop_gradient=self.stop_gradient
        return y

    def clamp_(self,min,max):
        paddorch.copy(fluid.layers.clip(self,float(min),float(max) ),self)
        # self.set_value( fluid.layers.clip(self,float(min),float(max) ) )
        return self

    def float(self):
        return convertTensor(self.astype('float32'))
    def long(self):
        return convertTensor(self.astype('int64'))

    def dot(self,x):
        return torch.dot(self,x)
    def add_(self,x):
        paddorch.copy(x+self,self)
        # self.set_value(x+self)
        return self
    def matmul(self,y):
        return torch.matmul(self,y)

    def norm(self,p=2,dim=-1, keepdim=True):
        return torch.norm(self,p=p,dim=dim,keepdim=keepdim)

    def expand(self,*sizes):
        if isinstance(sizes[0],Iterable):
            sizes=sizes[0]
        ##handle -1 case
        if len(sizes)>len(self.shape):
            for _ in range(len(sizes)-len(self.shape)):
                self=self.unsqueeze(dim=0)
        expand_times=[ x//y if x>=y else 1 for x,y in zip(sizes,self.shape) ]
        x= varbase_to_tensor(paddle.fluid.layers.expand(self, expand_times, name=None))
        return x

    def div_(self,x):
        # self.set_value(self/x)
        paddorch.copy(self/x, self)
        return  self

    def copy_(self,src):
        paddorch.copy(src,self)
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
        perm=[ len(perm)+x if x<0 else x for x in perm] ##not allow negative values
        x=paddle.transpose(self,perm)

        return varbase_to_tensor(x)

    def transpose(self,*perm):
        # if len(perm)==2 and len(self.shape)>2:
        if isinstance(perm[0],Iterable):
            return paddle.transpose(self,perm[0])
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
        if "dtype" in kwargs:
            dtype=str(kwargs["dtype"])
        elif isinstance(args[0],paddle.Tensor):
            dtype=str(args[0].dtype)
            if "64" in dtype:
                dtype="int32"
            elif "32" in dtype:
                dtype = "float32"
            else:
                return self
        else:
            dtype=str(args[0])
        if dtype=="int32":
            return self.long()
        elif dtype=="float32":
            return self.float()
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
                    return key.astype("int32")
            if isinstance(key,int):
                return paddorch.LongTensor(np.array([key]))
            if isinstance(key,list):
                key = paddorch.from_numpy(key).long()
            return key

        if isinstance(key, np.ndarray) or isinstance(key,paddle.Tensor):
            key = convert_key_to_inttensor(key)
        elif isinstance(key,Iterable) :
            if isinstance(key[0],slice):
                return super(Tensor, self).__setitem__(key, value)
            key2=[]
            for i in range(len(key)):
                key2.append(convert_key_to_inttensor(key[i]))

            key=paddle.stack(key2,axis=1)

            if len(key2)==1:
                key= key.reshape([-1])

        else:
            key=convert_key_to_inttensor(key)

        if key.shape[0]==0: ##empty selection, do nothing
            return self
        if not isinstance(value,paddle.Tensor):
            value=paddle.ones_like(key)*float(value)
        return paddle.scatter_(self,key,value)


    def __getitem__(self,args):
        from typing import   Iterable
        #
        # if isinstance(args, np.ndarray):
        #     # print(max(args),min(args),self.shape,len(args),len(set(args)) )
        #     args=paddorch.from_numpy(args).long()
        #     # print("converted numpy", type(args))
        # if  isinstance(args,paddle.Tensor):
        #     if args.dtype==paddle.fluid.core.VarDesc.VarType.BOOL:
        #         return convertTensor(paddle.masked_select(self,args))
        #     elif args.dtype==paddle.fluid.core.VarDesc.VarType.INT32 or args.dtype==paddle.fluid.core.VarDesc.VarType.INT64:
        #         return convertTensor(paddle.index_select(self,args,axis=0))
        # if isinstance(args,Iterable) and (len(args)==2):
        #     if isinstance(args[1],paddle.Tensor) and (args[1].dtype==paddle.fluid.core.VarDesc.VarType.BOOL):
        #         sel_indices=paddle.masked_select(paddle.arange(len(args[1])),args[1])  #paddle.arange(len(args[1]))[args[1]]
        #         return convertTensor(paddle.index_select(self[args[0]],sel_indices,axis=1))
        #     elif isinstance(args[0],paddle.Tensor):
        #             return convertTensor(super(Tensor, self).__getitem__(args[0])[args[1]])
        ##handle case using None to expand dimension
        if isinstance(args,Iterable):
            args2=list(args)
            for j in range(len(args)):
                k=len(args)-j-1
                if args[k] is None:
                    self.unsqueeze_(axis=k)
                    args2[k]=slice(None,None,None)
            args=tuple(args2)
        if getattr(self,'__getitem__origin',None) is None:
            return convertTensor(super(Tensor, self).__getitem__(args))
        else:
            return  self.__getitem__origin(args)



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
            grad.set_value(gradient+grad)
            return grad
        if gradient is not None:
            try:##only work in the dev version
                helper=self.register_hook(set_grad)
            except:
                pass
        if getattr(self,"backward_orig",None) is None:
            ret = super(Tensor, self).backward(retain_graph=retain_graph)
        else:
            ret= self.backward_orig(retain_graph=retain_graph)


        if gradient is not None:
            try:  ##only work in the dev version
                helper.remove()
            except:
                pass
        return ret

    @property
    def shape(self):
        # shape=paddle.shape(self)
        # if not isinstance(self,paddorch.Tensor):
        #     return shape
        if getattr(self,"shape_orig",None) is None:
            shape = super(Tensor, self).shape
        else:
            shape= self.shape_orig
            if isinstance(self,paddle.fluid.framework.ParamBase):
                return shape
        if isinstance(shape,int):
            return tuple(shape)
        if isinstance(shape[0],Iterable):
            shape=shape[0]
        return tuple( shape)


    @property
    def grad(self):
        if getattr(self,"grad_orig",None) is None:
            if super(Tensor, self).grad is None:
                return None
            return convertTensor(super(Tensor, self).grad)
        else:
            return self.grad_orig

    # def get_tensor(self):
    #     if self.stop_gradient:
    #         orig_stop_gradient=self.stop_gradient
    #         self.stop_gradient=False
    #         ret=self.float()
    #         ret= super(Tensor,ret).get_tensor()
    #         self.stop_gradient =orig_stop_gradient
    #     else:
    #         return super(Tensor,self).get_tensor()
    #     return ret

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
        index=index.astype("int32")

        ret= convertTensor(paddle.index_select(self,index=index, axis=dim))
        return  ret

    def masked_fill_(self, mask,value):
        mask=paddle.expand_as(mask,self)
        new_values=paddle.where(mask,self,paddle.ones(self.shape)*value)
        paddorch.copy(new_values,self)

        return self
    def masked_fill(self, mask,value):
        mask_float= mask.astype("float32")
        result = self *(1-mask_float) + mask_float*value
        return  result

        mask=paddle.expand_as(mask,self)
        new_values=paddle.where(mask,self,paddle.ones(self.shape)*value)
        return new_values

    def argmax(self,dim=0,keepdim=False):
        return convertTensor(paddle.argmax(self,axis=dim,keepdim=keepdim))

    def tolist(self):
        return self.cpu().numpy().tolist()
    def uniform_(self,low,high):
        paddorch.copy(paddorch.uniform_(self.shape,low,high),self)
        return self


    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        state['device']=str(state['device'])

        state['value']=self.cpu().numpy()
        state['dtype'] = str(state['value'].dtype)
        return state

    def __setstate__(self, state):
        # Restore instance attributes (i.e., filename and lineno).
        self.__init__(paddle.to_tensor(state['value'],dtype=state['dtype'] )  )

    def byte(self):
        return  self.astype("int32") ##paddle not support uin18

    def bernoulli_(self,p):
        paddorch.copy(paddle.bernoulli(paddle.ones_like(self)*p, name=None), self)
        return self

    def bool(self):
        return self.astype("bool")

    def chunk(self, chunks, dim=0 ):
        return paddle.chunk(self,chunks,axis=dim)
        # return  super(Tensor, self).chunk(chunks,axis=dim)


    def __invert__(self):
        return paddle.logical_not(self)

    def split(x, num_or_sections, dim=0):
        return torch.split(x, num_or_sections, dim)

    @staticmethod
    def new_tensor(self,*args,**kwargs):
        return paddle.to_tensor(*args,**kwargs)
        # np_arr=np.asarray(args[0])
        # dtype=str(np_arr.dtype).split(".")[-1]
        # kwargs['dtype']=dtype
        # return  paddorch.Tensor(*args,**kwargs).astype(dtype)

    def type_as(self,x):
        return self.astype(x.dtype)

    def __or__(self, other):
        return convertTensor(paddle.logical_or(self,other))


    def ne(self,x):
        return convertTensor( self!=x )


    def int(self):
        return  convertTensor(self.astype("int32"))

    def triu(self,diagonal=0):
        return convertTensor(paddorch.triu(self,diagonal=diagonal))

    def fill_diagonal(self,value,wrap=False):
        diag_v = paddle.diag(self)
        diag_v = torch.mm(paddle.eye(self.shape[0], self.shape[1]),
                          paddle.expand_as(diag_v, self)+value)
        return  self-diag_v