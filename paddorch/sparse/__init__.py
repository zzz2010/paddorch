import paddle
import paddorch

class FloatTensor(object):
    def __init__(self,indices,values,size):
        self.values=values
        self.indices=indices
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

def mm(sparseX:FloatTensor,denseY:paddle.fluid.dygraph.core.VarBase):
    ret_Mat=paddorch.zeros(sparseX.shape[0],denseY.shape[1])
    for i in range(sparseX._nnz()):
        row=int(sparseX.indices[0][i])
        col=int(sparseX.indices[1][i])
        ret_Mat[row]+=denseY[col]*sparseX.values[i]
    return ret_Mat
