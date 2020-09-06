from collections import OrderedDict

def load_pytorch_pretrain_model(paddle_model, pytorch_state_dict):
    '''
    paddle_model: dygraph layer object
    pytorch_state_dict: pytorch state_dict, assume in CPU device
    '''

    paddle_weight=paddle_model.state_dict()
    print("paddle num_params:",len(paddle_weight))
    print("torch num_params:", len(pytorch_state_dict))
    new_weight_dict=OrderedDict()
    torch_key_list=[]
    for key in pytorch_state_dict.keys():
        if "num_batches_tracked" in key:
            continue
        torch_key_list.append(key)

    for torch_key, paddle_key in zip(torch_key_list,paddle_weight.keys()):
        print(torch_key, paddle_key, pytorch_state_dict[torch_key].shape,paddle_weight[paddle_key].shape)
        if len(pytorch_state_dict[torch_key].shape)==0:
            continue
        ##handle all FC weight cases
        if ("fc" in torch_key and "weight" in torch_key) or (len(pytorch_state_dict[torch_key].shape)==2 and pytorch_state_dict[torch_key].shape[0]==pytorch_state_dict[torch_key].shape[1]):
            new_weight_dict[paddle_key] = pytorch_state_dict[torch_key].detach().numpy().T.astype("float32")
        elif int(paddle_weight[paddle_key].shape[-1])==int(pytorch_state_dict[torch_key].shape[-1])  :
            new_weight_dict[paddle_key]=pytorch_state_dict[torch_key].detach().numpy().astype("float32")
        else:
            new_weight_dict[paddle_key] = pytorch_state_dict[torch_key].detach().numpy().T.astype("float32")
    paddle_model.set_dict(new_weight_dict)
    return paddle_model.state_dict()


