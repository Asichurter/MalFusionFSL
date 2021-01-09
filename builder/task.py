import config

from comp.task import *

def buildTask(dataset,
              task_params: config.TaskConfig = None,
              model_params: config.ParamsConfig = None,
              optimize_params: config.OptimizeConfig = None,
              task_type='Train'):

    if model_params is None:
        model_params = config.params
    if optimize_params is None:
        optimize_params = config.optimize
    if task_params is None:
        task_params = config.task

    # TODO: 根据model_name来返回对应的task
    return _regular_task(dataset, task_params,
                         model_params, optimize_params,
                         task_type)


def _regular_task(dataset,
                  task_params: config.TaskConfig,
                  model_params: config.ParamsConfig,
                  optimize_params: config.OptimizeConfig,
                  task_type):
    expand = (optimize_params.LossFunc != 'nll')

    # TODO：默认不使用data_parallel
    return RegularEpisodeTask(k=task_params.Episode.k, qk=task_params.Episode.qk,
                              n=task_params.Episode.n, N=task_params.N,
                              dataset=dataset, cuda=True,
                              label_expand=expand, parallel=None,
                              task_type=task_type)