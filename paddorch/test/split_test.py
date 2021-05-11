from benchmarkbase import TorchBenchmarkBase
import paddorch as torch
from benchmark_utils import *


"""Microbenchmarks for Split operator"""


# Configs for PT Split operator
split_configs_short =  config_list(
    attr_names=["M", "N", "parts"],
    attrs=[
        [8, 8, 2],
        [256, 512, 2],
        [512, 512, 2],
    ],
    cross_product_configs={
        'device': ['cpu', 'cuda'],
    },
    tags=["short"],
)

split_configs_long = cross_product_configs(
    M=[128, 1024],
    N=[128, 1024],
    parts=[2, 4],
    device=['cpu', 'cuda'],
    tags=['long']
)



if __name__ == "__main__":
    for params in (split_configs_short + split_configs_long):
        p_dict={}
        for p in params:
            p_dict.update(p)
        M=p_dict['M']
        N=p_dict['N']
        parts=p_dict['parts']
        inputs = {
            "input": torch.rand(M, N ),
            "split_size": int(M * N / parts)
        }
        print(p_dict)

        import torch as th
        th_input=th.from_numpy(inputs['input'].numpy())
        torch_out=[x.shape for x in th.split(th_input,inputs['split_size'])]
        print("torch:",[x.shape for x in th.split(th_input,inputs['split_size'])])
        paddle_out=[x.shape for x in torch.split(inputs['input'], inputs['split_size'])]
        print("paddorch:",paddle_out)
        assert  torch_out==paddle_out

    M=100
    N=20
    parts=40
    inputs = {
        "input": torch.rand(M, N),
        "split_size": int(M * N / parts)
    }
    import torch as th
    th_input=th.from_numpy(inputs['input'].numpy())
    torch_out=[x.shape for x in th.split(th_input,inputs['split_size'])]
    print("torch:",[x.shape for x in th.split(th_input,inputs['split_size'])])
    paddle_out=[x.shape for x in torch.split(inputs['input'], inputs['split_size'])]
    print("paddorch:",paddle_out)
    assert  torch_out==paddle_out


    M=100
    N=20
    parts=40
    inputs = {
        "input": torch.rand(M, N),
        "split_size": int(M * N / parts)
    }
    inputs['split_size']=[37,63]
    import torch as th
    th_input=th.from_numpy(inputs['input'].numpy())
    torch_out=[x.shape for x in th.split(th_input,inputs['split_size'])]
    print("torch:",[x.shape for x in th.split(th_input,inputs['split_size'])])
    paddle_out=[x.shape for x in torch.split(inputs['input'], inputs['split_size'])]
    print("paddorch:",paddle_out)
    assert  torch_out==paddle_out