import paddle
import numpy as np
import paddorch

attn_prob=paddle.to_tensor(np.random.rand(100, 16, 9, 9)).astype("float32")
v_head=paddle.to_tensor(np.random.rand(100, 9, 16, 64)).astype("float32")
attn_prob.stop_gradient=False
v_head.stop_gradient=False
attention_dropout=paddorch.nn.Dropout(0.1)
attn_prob =attention_dropout(attn_prob)

# attention output, shape batch_size x seq_len x n_head x d_head
attn_vec = paddorch.einsum("bnij,bjnd->bind", attn_prob, v_head)



attn_vec.sum().backward()
print(attn_vec.grad)

