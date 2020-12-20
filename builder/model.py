import config
from utils.manager import PathManager

from model.ProtoNet import ProtoNet


def buildModel(path_manager: PathManager,
               model_params: config.ParamsConfig = None):  # 如果不使用默认的模型参数，则在此处给定一个ParamsConfig对象
   if model_params is None:
      model_params = config.params

   try:
      return ModelSwitch[model_params.ModelName](path_manager, model_params)
   except KeyError:
      raise ValueError("[ModelBuilder] No matched model implementation for '%s'"
                       % model_params.ModelName)

def _ProtoNet(path_manager: PathManager,
              model_params: config.ParamsConfig):
   return ProtoNet(model_params, path_manager)

ModelSwitch = {
   'ProtoNet': _ProtoNet
}