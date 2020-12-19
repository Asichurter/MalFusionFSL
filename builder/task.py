import config

from comp.task import *

def buildTask(dataset,
              task_params: config.TaskConfig = None,
              model_params: config.ParamsConfig = None,
              optimize_params: config.OptimizeConfig = None):

    if model_params is None:
        model_params = config.params
    if optimize_params is None:
        optimize_params = config.optimize
    if task_params is None:
        task_params = config.task

    # TODO: 根据model_name来返回对应的task
    return _regular(dataset, task_params,
                    model_params, optimize_params)


def _regular(dataset,
             task_params: config.TaskConfig = None,
             model_params: config.ParamsConfig = None,
             optimize_params: config.OptimizeConfig = None):
    expand = (optimize_params.LossFunc != 'nll')

    # TODO：默认不使用data_parallel
    return RegularEpisodeTask(k=task_params.k, qk=task_params.qk,
                              n=task_params.n, N=task_params.N,
                              dataset=dataset, cuda=True,
                              label_expand=expand, parallel=None)