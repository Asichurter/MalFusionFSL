import torch

import config


def _getLrFromCustomDict(name, dic):
    for n,l in dic.items():
        if name.startswith(n):      # 允许指定parameter的prefix来指定一个module中的所有参数的lr
            return False, l
    return True, None


def buildOptimizer(named_parameters, optimize_params=None):
    if optimize_params is None:
        optimize_params = config.optimize

    parameters = []
    default_pars_group = []
    for name, par in named_parameters:
        is_default, lr = _getLrFromCustomDict(name, optimize_params.CustomLrs)
        if is_default:
            default_pars_group.append(par)
        else:
            # 添加自定义学习率参数group
            # 如果lr自定义指定为None，则这一部分不作为参数进行训练
            if lr is not None:
                parameters.append({
                    'params': [par],
                    'lr': lr
                })

    # 添加默认学习率参数group
    parameters.append({
        'params': default_pars_group,
        'lr': optimize_params.DefaultLr
    })

    return OptimizerSwitch.get(optimize_params.Optimizer, _sgd)(parameters, optimize_params)

def _sgd(params, sgd_params: config.OptimizeConfig):
    sgd = torch.optim.SGD(params,
                          momentum=0.9,
                          weight_decay=sgd_params.WeightDecay)
    return sgd

def _adam(params, adam_params: config.OptimizeConfig):
    adam = torch.optim.Adam(params, weight_decay=adam_params.WeightDecay)
    return adam


OptimizerSwitch = {
    'sgd': _sgd,
    'adam': _adam,
}