import platform

from utils.file import loadJson
from utils.os import joinPath

# 所有可能的启动路径的config文件相对路径
_REL_CFG_PATHS = [
    "../config/",
    "config/"
]

# models using fast adaption
ADAPTED_MODELS = ['MetaSGD', 'ATAML', 'PerLayerATAML']

# models using multi-prototype
IMP_MODELS = ['IMP', 'SIMPLE', 'HybridIMP']


class EnvConfig:
    def __init__(self, cfg):
        self.DatasetBasePath = cfg['DatasetBasePath']
        self.ReportPath = cfg['ReportPath']

class EpisodeTaskConfig:
    def __init__(self, k=None, n=None, qk=None):
        self.k = k
        self.n = n
        self.qk = qk

class TaskConfig:
    def __init__(self, cfg=None):
        k = cfg['task']['k']
        n = cfg['task']['n']
        qk = cfg['task']['qk']
        self.epsiode = EpisodeTaskConfig(k, n, qk)

        self.dataset = cfg['task']['dataset']
        Ns = _loadJsonConfig(file_name="dataset_cap.json",
                             err_msg="没有合适的cap config文件相对路径")
        self.N = Ns[cfg['task']['dataset']]
        self.version = cfg['task']['version']
        self.deviceId =cfg['task']['device_id']

class TrainingConfig:
    def __init__(self, cfg):
        self.TrainEpoch = cfg['training']['epoch']
        self.ValCycle = cfg['validate']['val_cycle']
        self.ValEpisode = cfg['validate']['val_episode']
        self.Criteria = cfg['validate']['early_stop']['criteria']
        self.SaveLatest = cfg['validate']['early_stop']['save_latest']
        self.Desc = cfg['description']

class OptimizeConfig:
    def __init__(self, cfg):
        self.LossFunc = cfg['optimize']['loss_func']
        self.Optimizer = cfg['optimize']['optimizer']
        self.DefaultLr = cfg['optimize']['default_lr']
        self.CustomLrs = cfg['optimize']['custom_lrs']
        self.WeightDecay = cfg['optimize']['weight_decay']
        self.TaskBatch = cfg['optimize']['task_batch']
        self.Scheduler = cfg['optimize']['scheduler']       # 包含衰减周期和衰减倍率两个参数

class ParamsConfig:
    def __init__(self, cfg):
        self.ModelName = cfg['model']['model_name']
        self.Embedding = cfg['model']['embedding']
        self.SeqBackbone = cfg['model']['sequence_backbone']
        self.ConvBackbone = cfg['model']['conv_backbone']
        self.Regularization = cfg['model']['regularization']
        self.DataParallel = cfg['model']['data_parallel']
        self.ModelFeture = cfg['model']['model_feature']
        self.Cluster = cfg['model']['cluster']
        self.More = cfg['model']['more']

class VisConfig:
    def __init__(self, cfg):
        self.UseVisdom = cfg['visualization']['use_visdom']
        self.RecordGrad = cfg['visualization']['record_gradient']
        self.GradUpdateCycle = cfg['visualization']['gradient_update_cycle']
        self.PlotTypes = cfg['visualization']['plot']['types']
        self.PlotTitles = cfg['visualization']['plot']['titles']
        self.PlotXLabels = cfg['visualization']['plot']['xlabels']
        self.PlotYLabels = cfg['visualization']['plot']['ylabels']
        self.PlotLegends = cfg['visualization']['plot']['legends']

def _loadJsonConfig(file_name, err_msg):
    for rel_path in _REL_CFG_PATHS:
        try:
            cfg = loadJson(rel_path+file_name)
            return cfg
        except:
            continue
    raise RuntimeError("[ConfigInit] " + err_msg)

def _loadEnv():
    system_node = platform.node()
    env_cfg = _loadJsonConfig(file_name='env.json',
                              err_msg="没有合适的env config文件相对路径")
    return env_cfg['platform-node'][system_node]


env = EnvConfig(_loadEnv())
run_cfg = _loadJsonConfig(file_name="run.json",
                          err_msg="没有合适的run config文件相对路径")
task = TaskConfig(run_cfg)
train = TrainingConfig(run_cfg)
optimize = OptimizeConfig(run_cfg)
params = ParamsConfig(run_cfg)
vis = VisConfig(run_cfg)

__all__ = [env, task, train, optimize, params, vis,
           ADAPTED_MODELS, IMP_MODELS]

