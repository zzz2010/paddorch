import paddle
import paddorch
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds, eigs
class Tensor(object):
    def __init__(self,indices,values,size,use_row_split=False):
        ##make sure the row sorted order
        if use_row_split:
            order_ind=paddle.argsort(indices[0])
            self.values=paddorch.convertTensor(paddle.index_select(values,order_ind))
            self.indices=paddorch.convertTensor(paddle.index_select(indices,order_ind,axis=1))

                    ##record row start indices, and row end indices
            self.row_starts=[0]
            self.row_end = []
            last_row=0
            for ii,ind in enumerate(self.indices[0].numpy()):
                if last_row!=ind:
                    if ind-last_row>1: ##
                        for _ in range(ind-last_row):
                            self.row_starts.append(ii)
                            self.row_end.append( ii)
                    self.row_starts.append(ii)
                    self.row_end.append(ii)
                last_row=ind

            self.row_end.append(ii+1)
        else:
            self.values=values
            self.indices= indices
        self.indices.stop_gradient = True
        self.shape=size
        self.device=None

        sp_matrix=coo_matrix((self.values.numpy(), (self.indices[0].numpy(), self.indices[1].numpy())), shape=self.shape)
        u, s, vt = svds(sp_matrix, k=min(50,min(self.shape)-1))
        self.svd_left=paddorch.Tensor(u*s.reshape(1,-1))
        self.svd_left.stop_gradient=True
        self.svd_right = paddorch.Tensor(vt)
        self.svd_right.stop_gradient = True



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

    def to_dense(self):
        ret_mat=paddle.scatter_nd(paddle.transpose(self.indices,(1,0)),self.values,self.shape)
        return ret_mat

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

    updates=paddle.index_select(denseY, sparseX.indices[1] ,axis=0)*sparseX.values.view(-1,1)
    ret_Mat2=paddle.scatter_(ret_Mat,sparseX.indices[0] ,updates,overwrite=False)
    del ret_Mat
    ret_Mat2.stop_gradient=False ##re-enable gradient!
    return ret_Mat2


def mm(sparseX:FloatTensor,denseY:paddle.fluid.dygraph.core.VarBase,fast=False ):
    if fast:
        return  mm_fast(sparseX,denseY)
    else:
        # return  mm_backwardcorrect(sparseX,denseY)
        return (mm_svd(sparseX, denseY)+mm_fast(sparseX,denseY))/2

#new mm
def mm_splitrows(sparseX:FloatTensor,denseY:paddle.fluid.dygraph.core.VarBase,maxsize=557573 ): #55757376
    row_batch=max(1,min(sparseX.shape[0],maxsize//sparseX.shape[1]))
    mat_list=[]
    for i in range(sparseX.shape[0]//row_batch+1):
        if row_batch*i>=sparseX.shape[0]:
            break
        st=sparseX.row_starts[row_batch*i]
        if row_batch*(i+1)<len(sparseX.row_starts):
            ed=sparseX.row_end[row_batch*(i+1)-1]
        else:
            ed=sparseX.row_end[-1]

        n_row=min(row_batch,sparseX.shape[0]-row_batch*i)
        x_ind=sparseX.indices[0, st:ed]-row_batch*i
        y_ind=sparseX.indices[1,st:ed]
        sel_ind=paddle.stack([x_ind,y_ind],axis=1)

        mat = paddle.scatter_nd(sel_ind, sparseX.values[st:ed],(n_row, sparseX.shape[1]) )
        mat_list.append(paddle.mm(mat,denseY))
        del mat
    return  paddorch.cat(mat_list)


def mm_svd(sparseX:FloatTensor,denseY:paddle.fluid.dygraph.core.VarBase ):
    A=paddorch.mm(sparseX.svd_right,denseY)
    return paddorch.mm(sparseX.svd_left,A)


def mm_backwardcorrect(sparseX:FloatTensor,denseY:paddle.fluid.dygraph.core.VarBase ):


    update_inds=sparseX.indices[0]
    updates=paddle.index_select(denseY, sparseX.indices[1] ,axis=0)* sparseX.values .view(-1,1)
    ret_Mat2=paddle.scatter_nd( paddle.reshape(update_inds,(-1,1)) ,updates,(sparseX.shape[0],denseY.shape[1]))

    return ret_Mat2

def mm_backwardcorrect_slow(sparseX:FloatTensor,denseY:paddle.fluid.dygraph.core.VarBase ):
    ret_Mat = paddorch.zeros(sparseX.shape[0], denseY.shape[1])
    update_inds=sparseX.indices[0]
    updates=paddle.index_select(denseY, sparseX.indices[1] ,axis=0)* sparseX.values .view(-1,1)

    sorted_order=paddle.argsort(update_inds)

    update_inds_sorted=update_inds[sorted_order].detach().cpu().numpy()
    updates_sorted=paddle.index_select(updates,sorted_order)
    #use cpu loop
    row2range=dict()
    last_row=update_inds_sorted[0]
    start=0

    for i, row in enumerate(update_inds_sorted):
        if row!=last_row:
            row2range[last_row]=(start,i) #end is exclusive

            ret_Mat[int(row)]=paddle.sum(updates_sorted[int(start):int(i)],axis=0)

            start=i
            last_row=row

    ret_Mat[int(row)] = paddle.sum(updates_sorted[int(start):int(i)], axis=0)

    return ret_Mat




# def mm(sparseX:FloatTensor,denseY:paddle.fluid.dygraph.core.VarBase ):
#     ret_Mat=paddorch.zeros(sparseX.shape[0],denseY.shape[1])
#     ret_Mat.stop_gradient = True
#     updates=paddle.index_select(denseY, sparseX.indices[1] ,axis=0)*sparseX.values .view(-1,1)
#     ret_Mat2=paddle.scatter(ret_Mat,sparseX.indices[0] ,updates,overwrite=False)
#     return ret_Mat2