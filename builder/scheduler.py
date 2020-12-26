import torch

import config

class EmptyScheduler:
    def __init__(self):
        pass

    def step(self):
        # do nothing
        pass

def buildScheduler(optimizer,
                   optimize_params: config.OptimizeConfig = None):
    if optimize_params is None:
        optimize_params = config.optimize

    if optimize_params.Scheduler['type'] is None:
        return EmptyScheduler()

    return SchedulerSwitch.get(optimize_params.Scheduler['type'],
                               _empty_scheduler)(optimizer, optimize_params)

def _step_scheduler(optimizer,
                    optimize_params: config.OptimizeConfig):
    return torch.optim.lr_scheduler.StepLR(optimizer,
                                           step_size=optimize_params.Scheduler['lr_decay_iters'],
                                           gamma=optimize_params.Scheduler['lr_decay_gamma'])

def _empty_scheduler(optimizer,
                    optimize_params: config.OptimizeConfig):
    return EmptyScheduler()

SchedulerSwitch = {
    'step': _step_scheduler
}