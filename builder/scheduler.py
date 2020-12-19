import torch

import config


def buildScheduler(optimizer,
                   optimize_params: config.OptimizeConfig = None):
    if optimize_params is None:
        optimize_params = config.optimize

    # TODO: 增加其他优化器排程器的支持
    return _step_scheduler(optimizer,
                           optimize_params)

def _step_scheduler(optimizer,
                    optimize_params: config.OptimizeConfig = None):
    return torch.optim.lr_scheduler.StepLR(optimizer,
                                           step_size=optimize_params.Scheduler['lr_decay_iters'],
                                           gamma=optimize_params.Scheduler['lr_decay_gamma'])