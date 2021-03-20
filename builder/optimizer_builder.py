import torch

import config

def buildOptimizer(named_parameters, optimize_params=None):
    if optimize_params is None:
        optimize_params = config.optimize

    parameters = []
    for name, par in named_parameters:
        if name in optimize_params.CustomLrs:
            parameters += [{'params': [par], 'lr': optimize_params.CustomLrs[name]}]
        else:
            parameters += [{'params': [par], 'lr': optimize_params.DefaultLr}]

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