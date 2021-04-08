import torch
from pprint import pprint

import config
from utils.manager import PathManager
from model import *


def buildModel(path_manager: PathManager,
               task_config=None,
               model_params: config.ParamsConfig = None,
               loss_func=None,
               data_source=None,):
    if model_params is None:
        model_params = config.params

    if task_config is None:
        task_config = config.task

    try:
        model = ModelSwitch[model_params.ModelName](path_manager, model_params, task_config, loss_func, data_source)
    except KeyError:
        raise ValueError("[ModelBuilder] No matched model implementation for '%s'"
                         % model_params.ModelName)

    # 组装预训练的参数
    if len(task_config.PreloadStateDictVersions) > 0:
        remained_model_keys = [n for n,_ in model.named_parameters()]
        unexpected_keys = []
        for version in task_config.PreloadStateDictVersions:
            pm = PathManager(dataset=task_config.Dataset,
                             version=version,
                             model_name=model_params.ModelName)
            state_dict = torch.load(pm.model())
            load_result = model.load_state_dict(state_dict, strict=False)

            for k in state_dict.keys():
                if k not in load_result.unexpected_keys and k in remained_model_keys:
                    remained_model_keys.remove(k)

            unexpected_keys.extend(load_result.unexpected_keys)

        if len(remained_model_keys) > 0:
            print(f'[buildModel] Preloading, unloaded keys:')
            pprint(remained_model_keys)
        if len(unexpected_keys) > 0:
            print(f'[buildModel] Preloading, unexpected keys:')
            pprint(unexpected_keys)

    return model


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


def _SIMPLE(path_manager: PathManager,
            model_params: config.ParamsConfig,
            task_params: config.TaskConfig,
            loss_func,
            data_source):
    return SIMPLE(model_params, path_manager, loss_func, data_source).cuda()


def _PostProtoNet(path_manager: PathManager,
            model_params: config.ParamsConfig,
            task_params: config.TaskConfig,
            loss_func,
            data_source):
    return PostProtoNet(model_params, path_manager, loss_func, data_source).cuda()



ModelSwitch = {
    'ProtoNet': _ProtoNet,
    'NnNet': _NnNet,
    'HAPNet': _HAPNet,
    'SIMPLE': _SIMPLE,

    'PostProtoNet': _PostProtoNet,
}
