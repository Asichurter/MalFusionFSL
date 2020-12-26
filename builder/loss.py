from torch import nn

import config

def buildLossFunc(optimize_config: config.OptimizeConfig=None):
    if optimize_config is None:
        optimize_config = config.optimize

    return LossFuncSwitch.get(optimize_config.LossFunc, _nll)(optimize_config)

def _nll(optimize_config: config.OptimizeConfig):
    return nn.NLLLoss().cuda()      # 均默认使用cuda

def _mse(optimize_config: config.OptimizeConfig):
    return nn.MSELoss().cuda()

LossFuncSwitch = {
    'nll': _nll,
    'mse': _mse,
}