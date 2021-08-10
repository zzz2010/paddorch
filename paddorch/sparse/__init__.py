import paddle
import paddorch
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds, eigs
class Tensor(object):
    def __init__(self,indices,values,size,use_row_split=False,use_svd=False):

        self.values=values
        self.indices= indices.long()
        self.indices.stop_gradient = True
        self.shape=size
        self.device=""




    def _indices(self):
        return self.indices

    def _values(self):
        return self.values
    def _nnz(self):
        return len(self.indices[0])

    def to(self,*args, **kwargs):
        return self

    def __add__(self, other):
        paddorch.copy(self.values+other,self.values)
        # self.values.set_value(self.values+other)
        return  self


    def __mul__(self, other):
        paddorch.copy(self.values * other, self.values)

        # self.values.set_value(self.values*other)
        return  self


    def __div__(self, other_var):
        paddorch.copy(self.values / other_var, self.values)
        # self.values.set_value(self.values/other_var)
        return  self


    def coalesce(self):
        #merge the same index
        return  self #TODO

    def to_dense(self):
        ret_mat=paddle.scatter_nd(paddle.transpose(self.indices,(1,0)),self.values,self.shape)
        return ret_mat

class FloatTensor(Tensor):
    def __init__(self,indices,values,size):
        super(FloatTensor, self).__init__(indices,values,size)

class IntTensor(Tensor):
    def __init__(self,indices,values,size):
        super(IntTensor, self).__init__(indices,values.astype("int32"),size)



def mm(sparseX:FloatTensor,denseY:paddle.fluid.dygraph.core.VarBase  ):
    return mm_backwardcorrect_sparse_embed(sparseX, denseY)



def mm_backwardcorrect_sparse_embed(sparseX:FloatTensor,denseY:paddle.fluid.dygraph.core.VarBase ):
    update_inds=sparseX.indices[0]
    updates= paddle.nn.functional.embedding(sparseX.indices[1],denseY,sparse=False)* sparseX.values.view(-1,1)
    ret_Mat2=paddle.scatter_nd( paddle.reshape(update_inds,(-1,1)) ,updates,(sparseX.shape[0],denseY.shape[1]))

    return paddorch.convertTensor(ret_Mat2)
