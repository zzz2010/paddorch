import torch
def print_memory():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r - a  # free inside reserved
    print(a)

import paddorch
import time
import paddle
###see if branch significantly increase GPU memory

decide_var=1
N_size=7000
# print("dual branch")
# if decide_var>0:
#     for _ in range(10):
#         a=paddorch.randn((N_size,N_size))
#         time.sleep(1)
#         print_memory()
#
# else:
#     for _ in range(10):
#         a=paddorch.randn((N_size,N_size))
#         time.sleep(1)
#         print_memory()
#
# print("single branch")
# for _ in range(10):
#     paddorch.randn((N_size, N_size))
#     time.sleep(1)
#     print_memory()


# if decide_var>0:
#     for _ in range(10):
#         a=paddle.randn((N_size,N_size))
#         time.sleep(1)
#         print_memory()
#
# else:
#     for _ in range(10):
#         a=paddle.randn((N_size,N_size))
#         time.sleep(1)
#         print_memory()

print("single branch")
for _ in range(10):
    paddle.randn((N_size, N_size))
    time.sleep(1)
    print_memory()