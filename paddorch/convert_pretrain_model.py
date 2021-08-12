from collections import OrderedDict

def load_pytorch_pretrain_model(paddle_model, pytorch_state_dict ):
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
        torch_key_list.append(key )
    paddle_key_list = []
    for key in paddle_weight.keys():
        if ".cell" in key:
            continue
        paddle_key_list.append(key)
    paddle_unique_keys=set(paddle_key_list)-set(torch_key_list)
    print("paddle unique key , checking mis-alignment")
    print(paddle_unique_keys)
    for torch_key, paddle_key in zip(torch_key_list,paddle_key_list):

        print(torch_key, paddle_key, pytorch_state_dict[torch_key].shape,paddle_weight[paddle_key].shape)
        if len(pytorch_state_dict[torch_key].shape)==0:
            continue
        ##handle all FC weight cases
        if ("lin" in torch_key and "weight" in torch_key) or ("fc" in torch_key and "weight" in torch_key) or (len(pytorch_state_dict[torch_key].shape)==2 and (pytorch_state_dict[torch_key].shape[0]!=pytorch_state_dict[torch_key].shape[0]) and pytorch_state_dict[torch_key].shape[0]==pytorch_state_dict[torch_key].shape[1]):
            new_weight_dict[paddle_key] = pytorch_state_dict[torch_key].cpu().detach().numpy().T.astype("float32")
        elif int(paddle_weight[paddle_key].shape[-1])==int(pytorch_state_dict[torch_key].shape[-1])  :
            new_weight_dict[paddle_key]=pytorch_state_dict[torch_key].cpu().detach().numpy().astype("float32")
        else:
            new_weight_dict[paddle_key] = pytorch_state_dict[torch_key].cpu().detach().numpy().T.astype("float32")
    paddle_model.set_dict(new_weight_dict)
    return paddle_model.state_dict()



def load_pytorch_pretrain_model_remove_prefix(paddle_model, pytorch_state_dict,pytorch_prefix=""):
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
        torch_key_list.append(key.replace(pytorch_prefix,""))
    paddle_key_list = []
    for key in paddle_weight.keys():
        if ".cell" in key:
            continue
        paddle_key_list.append(key.replace(pytorch_prefix,""))
    torch_key_set=set(torch_key_list)
    paddle_key_set=set(paddle_key_list)
    paddle_unique_keys=paddle_key_set-torch_key_set
    print("paddle_unique_keys",paddle_unique_keys)
    missingkeys = torch_key_set - paddle_key_set
    print("torch_unique_keys", missingkeys)
    # _fast_init=True
    # if _fast_init:
    #     # retrieve unintialized modules and initialize
    #     missingkeys=torch_key_set-paddle_key_set
    #     print("torch unique key , checking mis-alignment")
    #     print(missingkeys)
    #     unintialized_modules = paddle_model.retrieve_modules_from_names(
    #         missingkeys, add_prefix="", remove_prefix=""
    #     )
    #     for module in unintialized_modules:
    #         paddle_model._init_weights(module)

    paddle_weight = paddle_model.state_dict()
    for torch_key in torch_key_set:
        # if "linears_prediction.4" not in paddle_key or "weight" not in paddle_key:
        #     continue
        paddle_key=torch_key
        if pytorch_prefix+paddle_key in paddle_weight:
            paddle_key=pytorch_prefix+paddle_key
        if paddle_key not in paddle_weight:
            continue
        if pytorch_prefix+torch_key in pytorch_state_dict:
            torch_key=pytorch_prefix+torch_key

        # print(torch_key, paddle_key, pytorch_state_dict[torch_key].shape,paddle_weight[paddle_key].shape)
        if len(pytorch_state_dict[torch_key].shape)==0:
            continue
        ##handle all FC weight cases
        if (  ("weight" in torch_key and "embed" not in torch_key and "conv" not in torch_key) and (len(pytorch_state_dict[torch_key].shape)==2) and   (pytorch_state_dict[torch_key].shape[0]==pytorch_state_dict[torch_key].shape[1]) ) or (len(pytorch_state_dict[torch_key].shape)==2 and (pytorch_state_dict[torch_key].shape[0]!=pytorch_state_dict[torch_key].shape[0]) and pytorch_state_dict[torch_key].shape[0]==pytorch_state_dict[torch_key].shape[1]):
            new_weight_dict[paddle_key] = pytorch_state_dict[torch_key].cpu().detach().numpy().T.astype("float32")
        elif int(paddle_weight[paddle_key].shape[-1])==int(pytorch_state_dict[torch_key].shape[-1])  :
            new_weight_dict[paddle_key]=pytorch_state_dict[torch_key].cpu().detach().numpy().astype("float32")
        else:
            new_weight_dict[paddle_key] = pytorch_state_dict[torch_key].cpu().detach().numpy().T.astype("float32")
        del pytorch_state_dict[torch_key] ##save memory
    paddle_model.set_dict(new_weight_dict)
    del new_weight_dict
    return paddle_model.state_dict()
