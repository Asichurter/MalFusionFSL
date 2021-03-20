import config
from utils.manager import PathManager

from model import *


def buildModel(path_manager: PathManager,
               task_config=None,
               model_params: config.ParamsConfig = None,
               loss_func=None,
               data_source=None):
    if model_params is None:
        model_params = config.params

    if task_config is None:
        task_config = config.task

    try:
        return ModelSwitch[model_params.ModelName](path_manager, model_params, task_config, loss_func, data_source)
    except KeyError:
        raise ValueError("[ModelBuilder] No matched model implementation for '%s'"
                         % model_params.ModelName)


def _ProtoNet(path_manager: PathManager,
              model_params: config.ParamsConfig,
              task_params: config.TaskConfig,
              loss_func,
              data_source):
    return ProtoNet(model_params, path_manager, loss_func, data_source).cuda()


def _NnNet(path_manager: PathManager,
           model_params: config.ParamsConfig,
           task_params: config.TaskConfig,
           loss_func,
           data_source):
    return NnNet(model_params, path_manager, loss_func, data_source).cuda()


def _HAPNet(path_manager: PathManager,
            model_params: config.ParamsConfig,
            task_params: config.TaskConfig,
            loss_func,
            data_source):
    return HAPNet(model_params, path_manager, loss_func, data_source, task_params.Episode.k).cuda()


ModelSwitch = {
    'ProtoNet': _ProtoNet,
    'NnNet': _NnNet,
    'HAPNet': _HAPNet
}
