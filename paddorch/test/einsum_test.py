import numpy as np
import paddorch
import paddle
import torch

def test_einsum(equnation,size1_size2):
    size1=size1_size2[0]
    size2=size1_size2[1]
    A = paddle.to_tensor(np.random.rand(*size1),dtype="float32",stop_gradient=False)
    B = paddle.to_tensor(np.random.rand(*size2),dtype="float32",stop_gradient=False)

    C=paddorch.einsum(equnation, A,B)
    C.sum().backward()
    print(equnation,size1_size2)
    paddle_grad=A.gradient()

    A=torch.from_numpy(A.numpy())
    A.requires_grad=True
    B=torch.from_numpy(B.numpy())
    B.requires_grad=True

    C=torch.einsum(equnation, A,B)
    C.sum().backward( )
    torch_grad=A.grad.numpy()
    grad_diff=torch.FloatTensor(torch_grad-paddle_grad).abs().max()
    print(float(grad_diff))
    assert  grad_diff<1e-3, "%.4f , %.4f"%(torch_grad.mean(),paddle_grad.mean())

    # print("A.grad",A.gradient().mean())
    # print("B.grad",A.gradient().mean())




test_cases="""operands type: bind,bjnd->bnij [(100, 9, 16, 64), (100, 9, 16, 64)]
operands type: td,dnh->tnh [(18, 1024), [1024, 16, 64]]
operands type: binh,tnh->bnit [(100, 9, 16, 64), (18, 16, 64)]
operands type: bind,snd->bnis [(100, 9, 16, 64), [2, 16, 64]]
operands type: bnij,bjnd->bind [(100, 16, 9, 9), (100, 9, 16, 64)]
operands type: bind,bjnd->bnij [(100, 9, 16, 64), (100, 9, 16, 64)]
operands type: td,dnh->tnh [(18, 1024), [1024, 16, 64]]
operands type: binh,tnh->bnit [(100, 9, 16, 64), (18, 16, 64)]
operands type: bind,snd->bnis [(100, 9, 16, 64), [2, 16, 64]]
operands type: bnij,bjnd->bind [(100, 16, 9, 9), (100, 9, 16, 64)]
operands type: bind,bjnd->bnij [(100, 9, 16, 64), (100, 9, 16, 64)]
operands type: td,dnh->tnh [(18, 1024), [1024, 16, 64]]
operands type: binh,tnh->bnit [(100, 9, 16, 64), (18, 16, 64)]
operands type: bind,snd->bnis [(100, 9, 16, 64), [2, 16, 64]]
operands type: bnij,bjnd->bind [(100, 16, 9, 9), (100, 9, 16, 64)]
operands type: bind,bjnd->bnij [(100, 9, 16, 64), (100, 9, 16, 64)]
operands type: td,dnh->tnh [(18, 1024), [1024, 16, 64]]
operands type: binh,tnh->bnit [(100, 9, 16, 64), (18, 16, 64)]
operands type: bind,snd->bnis [(100, 9, 16, 64), [2, 16, 64]]
operands type: bnij,bjnd->bind [(100, 16, 9, 9), (100, 9, 16, 64)]
operands type: bind,bjnd->bnij [(100, 9, 16, 64), (100, 9, 16, 64)]
operands type: td,dnh->tnh [(18, 1024), [1024, 16, 64]]
operands type: binh,tnh->bnit [(100, 9, 16, 64), (18, 16, 64)]
operands type: bind,snd->bnis [(100, 9, 16, 64), [2, 16, 64]]
operands type: bnij,bjnd->bind [(100, 16, 9, 9), (100, 9, 16, 64)]
operands type: bind,bjnd->bnij [(100, 9, 16, 64), (100, 9, 16, 64)]
operands type: td,dnh->tnh [(18, 1024), [1024, 16, 64]]
operands type: binh,tnh->bnit [(100, 9, 16, 64), (18, 16, 64)]
operands type: bind,snd->bnis [(100, 9, 16, 64), [2, 16, 64]]
operands type: bnij,bjnd->bind [(100, 16, 9, 9), (100, 9, 16, 64)]
operands type: bind,bjnd->bnij [(100, 9, 16, 64), (100, 9, 16, 64)]
operands type: td,dnh->tnh [(18, 1024), [1024, 16, 64]]
operands type: binh,tnh->bnit [(100, 9, 16, 64), (18, 16, 64)]
operands type: bind,snd->bnis [(100, 9, 16, 64), [2, 16, 64]]
operands type: bnij,bjnd->bind [(100, 16, 9, 9), (100, 9, 16, 64)]
operands type: bind,bjnd->bnij [(100, 9, 16, 64), (100, 9, 16, 64)]
operands type: td,dnh->tnh [(18, 1024), [1024, 16, 64]]
operands type: binh,tnh->bnit [(100, 9, 16, 64), (18, 16, 64)]
operands type: bind,snd->bnis [(100, 9, 16, 64), [2, 16, 64]]
operands type: bnij,bjnd->bind [(100, 16, 9, 9), (100, 9, 16, 64)]
operands type: bind,bjnd->bnij [(100, 9, 16, 64), (100, 9, 16, 64)]
operands type: td,dnh->tnh [(18, 1024), [1024, 16, 64]]
operands type: binh,tnh->bnit [(100, 9, 16, 64), (18, 16, 64)]
operands type: bind,snd->bnis [(100, 9, 16, 64), [2, 16, 64]]
operands type: bnij,bjnd->bind [(100, 16, 9, 9), (100, 9, 16, 64)]
operands type: bind,bjnd->bnij [(100, 9, 16, 64), (100, 9, 16, 64)]
operands type: td,dnh->tnh [(18, 1024), [1024, 16, 64]]
operands type: binh,tnh->bnit [(100, 9, 16, 64), (18, 16, 64)]
operands type: bind,snd->bnis [(100, 9, 16, 64), [2, 16, 64]]
operands type: bnij,bjnd->bind [(100, 16, 9, 9), (100, 9, 16, 64)]
operands type: bind,bjnd->bnij [(100, 5, 16, 64), (100, 9, 16, 64)]
operands type: td,dnh->tnh [(19, 1024), [1024, 16, 64]]
operands type: binh,tnh->bnit [(100, 5, 16, 64), (19, 16, 64)]
operands type: bind,snd->bnis [(100, 5, 16, 64), [2, 16, 64]]
operands type: bnij,bjnd->bind [(100, 16, 5, 9), (100, 9, 16, 64)]
operands type: bind,bjnd->bnij [(100, 5, 16, 64), (100, 5, 16, 64)]
operands type: td,dnh->tnh [(10, 1024), [1024, 16, 64]]
operands type: binh,tnh->bnit [(100, 5, 16, 64), (10, 16, 64)]
operands type: bind,snd->bnis [(100, 5, 16, 64), [2, 16, 64]]
operands type: bnij,bjnd->bind [(100, 16, 5, 5), (100, 5, 16, 64)]
operands type: bind,bjnd->bnij [(100, 5, 16, 64), (100, 5, 16, 64)]
operands type: td,dnh->tnh [(10, 1024), [1024, 16, 64]]
operands type: binh,tnh->bnit [(100, 5, 16, 64), (10, 16, 64)]
operands type: bind,snd->bnis [(100, 5, 16, 64), [2, 16, 64]]
operands type: bnij,bjnd->bind [(100, 16, 5, 5), (100, 5, 16, 64)]
operands type: bind,bjnd->bnij [(100, 5, 16, 64), (100, 5, 16, 64)]
operands type: td,dnh->tnh [(10, 1024), [1024, 16, 64]]
operands type: binh,tnh->bnit [(100, 5, 16, 64), (10, 16, 64)]
operands type: bind,snd->bnis [(100, 5, 16, 64), [2, 16, 64]]
operands type: bnij,bjnd->bind [(100, 16, 5, 5), (100, 5, 16, 64)]
operands type: bind,bjnd->bnij [(100, 5, 16, 64), (100, 5, 16, 64)]
operands type: td,dnh->tnh [(10, 1024), [1024, 16, 64]]
operands type: binh,tnh->bnit [(100, 5, 16, 64), (10, 16, 64)]
operands type: bind,snd->bnis [(100, 5, 16, 64), [2, 16, 64]]
operands type: bnij,bjnd->bind [(100, 16, 5, 5), (100, 5, 16, 64)]
operands type: bind,bjnd->bnij [(100, 5, 16, 64), (100, 5, 16, 64)]
operands type: td,dnh->tnh [(10, 1024), [1024, 16, 64]]
operands type: binh,tnh->bnit [(100, 5, 16, 64), (10, 16, 64)]
operands type: bind,snd->bnis [(100, 5, 16, 64), [2, 16, 64]]
operands type: bnij,bjnd->bind [(100, 16, 5, 5), (100, 5, 16, 64)]
operands type: bind,bjnd->bnij [(100, 5, 16, 64), (100, 5, 16, 64)]
operands type: td,dnh->tnh [(10, 1024), [1024, 16, 64]]
operands type: binh,tnh->bnit [(100, 5, 16, 64), (10, 16, 64)]
operands type: bind,snd->bnis [(100, 5, 16, 64), [2, 16, 64]]
operands type: bnij,bjnd->bind [(100, 16, 5, 5), (100, 5, 16, 64)]
operands type: bind,bjnd->bnij [(100, 5, 16, 64), (100, 5, 16, 64)]
operands type: td,dnh->tnh [(10, 1024), [1024, 16, 64]]
operands type: binh,tnh->bnit [(100, 5, 16, 64), (10, 16, 64)]
operands type: bind,snd->bnis [(100, 5, 16, 64), [2, 16, 64]]
operands type: bnij,bjnd->bind [(100, 16, 5, 5), (100, 5, 16, 64)]
operands type: bind,bjnd->bnij [(100, 5, 16, 64), (100, 5, 16, 64)]
operands type: td,dnh->tnh [(10, 1024), [1024, 16, 64]]
operands type: binh,tnh->bnit [(100, 5, 16, 64), (10, 16, 64)]
operands type: bind,snd->bnis [(100, 5, 16, 64), [2, 16, 64]]
operands type: bnij,bjnd->bind [(100, 16, 5, 5), (100, 5, 16, 64)]
operands type: bind,bjnd->bnij [(100, 5, 16, 64), (100, 5, 16, 64)]
operands type: td,dnh->tnh [(10, 1024), [1024, 16, 64]]
operands type: binh,tnh->bnit [(100, 5, 16, 64), (10, 16, 64)]
operands type: bind,snd->bnis [(100, 5, 16, 64), [2, 16, 64]]
operands type: bnij,bjnd->bind [(100, 16, 5, 5), (100, 5, 16, 64)]
operands type: bind,bjnd->bnij [(100, 3, 16, 64), (100, 5, 16, 64)]
operands type: td,dnh->tnh [(11, 1024), [1024, 16, 64]]
operands type: binh,tnh->bnit [(100, 3, 16, 64), (11, 16, 64)]
operands type: bind,snd->bnis [(100, 3, 16, 64), [2, 16, 64]]
operands type: bnij,bjnd->bind [(100, 16, 3, 5), (100, 5, 16, 64)]
operands type: bind,bjnd->bnij [(100, 3, 16, 64), (100, 3, 16, 64)]
operands type: td,dnh->tnh [(6, 1024), [1024, 16, 64]]
operands type: binh,tnh->bnit [(100, 3, 16, 64), (6, 16, 64)]
operands type: bind,snd->bnis [(100, 3, 16, 64), [2, 16, 64]]
operands type: bnij,bjnd->bind [(100, 16, 3, 3), (100, 3, 16, 64)]
operands type: bind,bjnd->bnij [(100, 3, 16, 64), (100, 3, 16, 64)]
operands type: td,dnh->tnh [(6, 1024), [1024, 16, 64]]
operands type: binh,tnh->bnit [(100, 3, 16, 64), (6, 16, 64)]
operands type: bind,snd->bnis [(100, 3, 16, 64), [2, 16, 64]]
operands type: bnij,bjnd->bind [(100, 16, 3, 3), (100, 3, 16, 64)]
operands type: bind,bjnd->bnij [(100, 3, 16, 64), (100, 3, 16, 64)]
operands type: td,dnh->tnh [(6, 1024), [1024, 16, 64]]
operands type: binh,tnh->bnit [(100, 3, 16, 64), (6, 16, 64)]
operands type: bind,snd->bnis [(100, 3, 16, 64), [2, 16, 64]]
operands type: bnij,bjnd->bind [(100, 16, 3, 3), (100, 3, 16, 64)]
operands type: bind,bjnd->bnij [(100, 3, 16, 64), (100, 3, 16, 64)]
operands type: td,dnh->tnh [(6, 1024), [1024, 16, 64]]
operands type: binh,tnh->bnit [(100, 3, 16, 64), (6, 16, 64)]
operands type: bind,snd->bnis [(100, 3, 16, 64), [2, 16, 64]]
operands type: bnij,bjnd->bind [(100, 16, 3, 3), (100, 3, 16, 64)]
operands type: bind,bjnd->bnij [(100, 3, 16, 64), (100, 3, 16, 64)]
operands type: td,dnh->tnh [(6, 1024), [1024, 16, 64]]
operands type: binh,tnh->bnit [(100, 3, 16, 64), (6, 16, 64)]
operands type: bind,snd->bnis [(100, 3, 16, 64), [2, 16, 64]]
operands type: bnij,bjnd->bind [(100, 16, 3, 3), (100, 3, 16, 64)]
operands type: bind,bjnd->bnij [(100, 3, 16, 64), (100, 3, 16, 64)]
operands type: td,dnh->tnh [(6, 1024), [1024, 16, 64]]
operands type: binh,tnh->bnit [(100, 3, 16, 64), (6, 16, 64)]
operands type: bind,snd->bnis [(100, 3, 16, 64), [2, 16, 64]]
operands type: bnij,bjnd->bind [(100, 16, 3, 3), (100, 3, 16, 64)]
operands type: bind,bjnd->bnij [(100, 3, 16, 64), (100, 3, 16, 64)]
operands type: td,dnh->tnh [(6, 1024), [1024, 16, 64]]
operands type: binh,tnh->bnit [(100, 3, 16, 64), (6, 16, 64)]
operands type: bind,snd->bnis [(100, 3, 16, 64), [2, 16, 64]]
operands type: bnij,bjnd->bind [(100, 16, 3, 3), (100, 3, 16, 64)]
operands type: bind,bjnd->bnij [(100, 3, 16, 64), (100, 3, 16, 64)]
operands type: td,dnh->tnh [(6, 1024), [1024, 16, 64]]
operands type: binh,tnh->bnit [(100, 3, 16, 64), (6, 16, 64)]
operands type: bind,snd->bnis [(100, 3, 16, 64), [2, 16, 64]]
operands type: bnij,bjnd->bind [(100, 16, 3, 3), (100, 3, 16, 64)]
operands type: bind,bjnd->bnij [(100, 3, 16, 64), (100, 3, 16, 64)]
operands type: td,dnh->tnh [(6, 1024), [1024, 16, 64]]
operands type: binh,tnh->bnit [(100, 3, 16, 64), (6, 16, 64)]
operands type: bind,snd->bnis [(100, 3, 16, 64), [2, 16, 64]]
operands type: bnij,bjnd->bind [(100, 16, 3, 3), (100, 3, 16, 64)]
operands type: bind,bjnd->bnij [(100, 9, 16, 64), (100, 9, 16, 64)]
operands type: td,dnh->tnh [(18, 1024), [1024, 16, 64]]
operands type: binh,tnh->bnit [(100, 9, 16, 64), (18, 16, 64)]
operands type: bind,snd->bnis [(100, 9, 16, 64), [2, 16, 64]]
operands type: bnij,bjnd->bind [(100, 16, 9, 9), (100, 9, 16, 64)]
operands type: bind,bjnd->bnij [(100, 9, 16, 64), (100, 9, 16, 64)]
operands type: td,dnh->tnh [(18, 1024), [1024, 16, 64]]
operands type: binh,tnh->bnit [(100, 9, 16, 64), (18, 16, 64)]
operands type: bind,snd->bnis [(100, 9, 16, 64), [2, 16, 64]]
operands type: bnij,bjnd->bind [(100, 16, 9, 9), (100, 9, 16, 64)]
"""

sum_test=0
for line in test_cases.split('\n'):
    comps=line.split(" ")
    if len(comps)<4:
        continue
    size1_size2=eval(" ".join(comps[3:]))
    eq_str=comps[2].strip()
    sum_test=test_einsum(eq_str,size1_size2)


# test_einsum('bind,bjnd->bnij', [(100, 9, 16, 64), (100, 9, 16, 64)])
#
#
# test_einsum('td,dnh->tnh' , [(18, 1024), [1024, 16, 64]])
#
#
# test_einsum('binh,tnh->bnit' , [(100, 9, 16, 64), (18, 16, 64)])
#
# test_einsum('bind,snd->bnis',   [(100, 9, 16, 64), [2, 16, 64]])
#
# test_einsum('bnij,bjnd->bind',  [(100, 16, 9, 9), (100, 9, 16, 64)])