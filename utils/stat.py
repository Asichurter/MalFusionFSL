import torch
import numpy as np

def statParamNumber(model: torch.nn.Module):
    num_of_params = 0
    param_dict = {}
    for name,par in model.named_parameters():
        num_of_params += par.numel()
        param_dict[name] = par.numel()
    print('params:', num_of_params)
    for k,v in param_dict.items():
        print(k, v)

def calBeliefeInterval(datas):
    '''
    计算数据95%的置信区间。依据的是t分布的95%置信区间公式
    '''
    # assert len(datas)%split == 0, '数据不可被split等分。数据长度=%d  split=%d'%(len(datas, split))

    # if type(datas) == list:
    #    datas = np.array(datas)
    # datas = datas.reshape(split,-1)
    # means = datas.mean(axis=1).reshape(-1)
    # std = np.std(means)

    z = 1.95996
    s = np.std(datas, ddof=1)
    n = len(datas)

    return z*s/np.sqrt(n)