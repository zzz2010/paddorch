import paddle
import paddorch

class Tensor(object):
    def __init__(self,indices,values,size):
        self.values=values
        self.indices=indices
        self.indices.stop_gradient = True
        self.shape=size
        self.device=None

    def _indices(self):
        return self.indices

    def _values(self):
        return self.values
    def _nnz(self):
        return len(self.indices[0])

    def to(self,*args, **kwargs):
        return self

    def __add__(self, other):
        self.values.set_value(self.values+other)
        return  self


    def __mul__(self, other):
        self.values.set_value(self.values*other)
        return  self


    def __div__(self, other_var):
        self.values.set_value(self.values/other_var)
        return  self


    def coalesce(self):
        #merge the same index
        return  self #TODO

class FloatTensor(Tensor):
    def __init__(self,indices,values,size):
        super(FloatTensor, self).__init__(indices,values,size)

class IntTensor(Tensor):
    def __init__(self,indices,values,size):
        super(IntTensor, self).__init__(indices,values.astype("int32"),size)


##slow version
# def mm(sparseX:FloatTensor,denseY:paddle.fluid.dygraph.core.VarBase):
#     ret_Mat=paddorch.zeros(sparseX.shape[0],denseY.shape[1])
#     for i in range(sparseX._nnz()):
#         row=int(sparseX.indices[0][i])
#         col=int(sparseX.indices[1][i])
#         ret_Mat[row]+=denseY[col]*sparseX.values[i]
#     return ret_Mat

def mm_smallmem(sparseX:FloatTensor,denseY:paddle.fluid.dygraph.core.VarBase,max_query_size=2000000):
    batch_size = sparseX._nnz()
     #avoid memory explode
    if batch_size>max_query_size:
        batch_size=max_query_size//sparseX.shape[1]*sparseX.shape[1]

    ret_Mat=paddorch.zeros(sparseX.shape[0],denseY.shape[1])
    for batch_i in range(sparseX._nnz()//batch_size):
        updates=paddle.index_select(denseY, sparseX.indices[1][(batch_i*batch_size):( (batch_i+1)*batch_size) ],axis=0)*sparseX.values[(batch_i*batch_size):( (batch_i+1)*batch_size) ].view(-1,1)
        ret_Mat=paddle.scatter_(ret_Mat,sparseX.indices[0][(batch_i*batch_size):( (batch_i+1)*batch_size) ],updates,overwrite=False)
    return ret_Mat


#old mm,  Use this one!! faster
def mm_fast(sparseX:FloatTensor,denseY:paddle.fluid.dygraph.core.VarBase ):

    ret_Mat=paddorch.zeros(sparseX.shape[0],denseY.shape[1])
    ret_Mat.stop_gradient=True

    updates=paddle.index_select(denseY, sparseX.indices[1] ,axis=0)*sparseX.values .view(-1,1)
    ret_Mat2=paddle.scatter_(ret_Mat,sparseX.indices[0] ,updates,overwrite=False)
    del ret_Mat
    ret_Mat2.stop_gradient=False ##re-enable gradient!
    return ret_Mat2


def mm(sparseX:FloatTensor,denseY:paddle.fluid.dygraph.core.VarBase,fast=True ):
    if fast:
        return  mm_fast(sparseX,denseY)
    else:
        return  mm_backwardcorrect(sparseX,denseY)

#new mm
def mm_backwardcorrect(sparseX:FloatTensor,denseY:paddle.fluid.dygraph.core.VarBase ):


    update_inds=sparseX.indices[0]
    updates=paddle.index_select(denseY, sparseX.indices[1] ,axis=0)*paddle.unsqueeze_(sparseX.values,1) #sparseX.values .view(-1,1)
    ret_Mat2=paddle.scatter_nd( paddle.reshape(update_inds,(-1,1)) ,updates,(sparseX.shape[0],denseY.shape[1]))

    return ret_Mat2

# def mm(sparseX:FloatTensor,denseY:paddle.fluid.dygraph.core.VarBase ):
#     ret_Mat=paddorch.zeros(sparseX.shape[0],denseY.shape[1])
#     ret_Mat.stop_gradient = True
#     updates=paddle.index_select(denseY, sparseX.indices[1] ,axis=0)*sparseX.values .view(-1,1)
#     ret_Mat2=paddle.scatter(ret_Mat,sparseX.indices[0] ,updates,overwrite=False)
#     return ret_Mat2