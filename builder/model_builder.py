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
        model = ModelSwitch[model_params.ModelName](path_manager, model_params, task_config, loss_func, data_source)\
                .cuda()
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
    return ProtoNet(model_params, path_manager, loss_func, data_source)


def _NnNet(path_manager: PathManager,
           model_params: config.ParamsConfig,
           task_params: config.TaskConfig,
           loss_func,
           data_source):
    return NnNet(model_params, path_manager, loss_func, data_source)


def _HAPNet(path_manager: PathManager,
            model_params: config.ParamsConfig,
            task_params: config.TaskConfig,
            loss_func,
            data_source):
    return HAPNet(model_params, path_manager, loss_func, data_source, task_params.Episode.k)


def _SIMPLE(path_manager: PathManager,
            model_params: config.ParamsConfig,
            task_params: config.TaskConfig,
            loss_func,
            data_source):
    return SIMPLE(model_params, path_manager, loss_func, data_source)


def _IMP(path_manager: PathManager,
         model_params: config.ParamsConfig,
         task_params: config.TaskConfig,
         loss_func,
         data_source):
    return IMP(model_params, path_manager, loss_func, data_source)


def _PostProtoNet(path_manager: PathManager,
                  model_params: config.ParamsConfig,
                  task_params: config.TaskConfig,
                  loss_func,
                  data_source):
    return PostProtoNet(model_params, path_manager, loss_func, data_source)


def _MLossProtoNet(path_manager: PathManager,
                   model_params: config.ParamsConfig,
                   task_params: config.TaskConfig,
                   loss_func,
                   data_source):
    return MLossProtoNet(model_params, path_manager, loss_func, data_source)


def _MLossSIMPLE(path_manager: PathManager,
                 model_params: config.ParamsConfig,
                 task_params: config.TaskConfig,
                 loss_func,
                 data_source):
    return MLossSIMPLE(model_params, path_manager, loss_func, data_source)


def _MLossIMP(path_manager: PathManager,
              model_params: config.ParamsConfig,
              task_params: config.TaskConfig,
              loss_func,
              data_source):
    return MLossIMP(model_params, path_manager, loss_func, data_source)


def _FEAT(path_manager: PathManager,
          model_params: config.ParamsConfig,
          task_params: config.TaskConfig,
          loss_func,
          data_source):
    return FEAT(model_params, path_manager, loss_func, data_source)


def _ConvProtoNet(path_manager: PathManager,
                  model_params: config.ParamsConfig,
                  task_params: config.TaskConfig,
                  loss_func,
                  data_source):
    return ConvProtoNet(model_params, path_manager, loss_func, data_source, task_params.Episode.k)


def _InductionNet(path_manager: PathManager,
                  model_params: config.ParamsConfig,
                  task_params: config.TaskConfig,
                  loss_func,
                  data_source):
    return InductionNet(model_params, path_manager, loss_func, data_source)


ModelSwitch = {
    'ProtoNet': _ProtoNet,
    'NnNet': _NnNet,
    'HAPNet': _HAPNet,
    'SIMPLE': _SIMPLE,
    'IMP': _IMP,
    'FEAT': _FEAT,
    'ConvProtoNet': _ConvProtoNet,
    'InductionNet': _InductionNet,

    'PostProtoNet': _PostProtoNet,

    'MLossProtoNet': _MLossProtoNet,
    "MLossSIMPLE": _MLossSIMPLE,
    'MLossIMP': _MLossIMP,
}
