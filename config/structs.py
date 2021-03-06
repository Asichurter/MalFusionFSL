import os

from utils.file import loadJson
from .const import *

def _loadJsonConfig(file_name, err_msg):
    for rel_path in REL_CFG_PATHS:
        try:
            cfg = loadJson(rel_path+file_name)
            return cfg
        except FileNotFoundError:
            print(f"[ConfigInit] not found: {rel_path+file_name}")
            continue
    raise RuntimeError(f"[ConfigInit] pwd: {os.getcwd()}, {err_msg}")

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
        self.Episode = EpisodeTaskConfig(k, n, qk)

        self.Dataset = cfg['task']['dataset']
        Ns = _loadJsonConfig(file_name="datcap.json",
                             err_msg="没有合适的cap config文件相对路径")
        self.N = Ns[cfg['task']['dataset']]
        self.Version = cfg['task']['version']
        self.DeviceId =cfg['task']['device_id']
        self.PreloadStateDictVersions = cfg['task'].get('preload_state_dict_versions', [])

class TrainingConfig:
    def __init__(self, cfg):             
        self.TrainEpoch = int(cfg['training']['epoch'])
        self.DataSource = cfg['training']['data_source']
        self.ValCycle = int(cfg['validate']['val_cycle'])
        self.ValEpisode = int(cfg['validate']['val_episode'])
        self.Criteria = cfg['validate']['early_stop']['criteria']
        self.SaveLatest = cfg['validate']['early_stop']['save_latest']
        self.Metrics = cfg['validate']['metrics']
        self.Desc = cfg['description']
        self.Verbose = cfg['training']['verbose']
        self.ClipGradNorm = cfg['training'].get('clip_grad_norm', None)     # 向下适配

class OptimizeConfig:
    def __init__(self, cfg):
        self.LossFunc = cfg['optimize']['loss_function']
        self.Optimizer = cfg['optimize']['optimizer']
        self.DefaultLr = cfg['optimize']['default_lr']
        self.CustomLrs = cfg['optimize']['custom_lrs']
        self.WeightDecay = cfg['optimize']['weight_decay']
        self.Momentum = cfg['optimize'].get('momentum', 0.9)    # 向下适配
        self.TaskBatch = cfg['optimize']['task_batch']
        self.Scheduler = cfg['optimize']['scheduler']       # 包含衰减周期和衰减倍率两个参数

class ParamsConfig:
    def __init__(self, cfg):
        self.ModelName = cfg['model']['model_name']
        self.Reproject = cfg['model']['reproject']
        # self.FeatureDim = cfg['model']['feature_dim']
        self.Fusion = cfg['model']['fusion']
        self.Embedding = cfg['model']['embedding']
        self.SeqBackbone = cfg['model']['sequence_backbone']
        self.ConvBackbone = cfg['model']['conv_backbone']
        self.Regularization = cfg['model']['regularization']
        self.DataParallel = cfg['model']['data_parallel']
        self.ModelFeture = cfg['model']['model_feature']
        self.Cluster = cfg['model']['cluster']
        self.More = cfg['model']['more']

class PlotConfig:
    def __init__(self, cfg):
        self.Enabled = cfg['visualization']['enabled']
        self.Type = cfg['visualization']['type']
        self.RecordGrad = cfg['visualization']['record_gradient']
        self.GradUpdateCycle = cfg['visualization']['gradient_update_cycle']
        self.PlotTypes = cfg['visualization']['plot']['types']
        self.PlotTitles = cfg['visualization']['plot']['titles']
        self.PlotXLabels = cfg['visualization']['plot']['xlabels']
        self.PlotYLabels = cfg['visualization']['plot']['ylabels']
        self.PlotLegends = cfg['visualization']['plot']['legends']

class TestConfig:
    def __init__(self, cfg):
        self.Task = TaskConfig(cfg)
        self.ModelName = cfg['model_name']
        self.Epoch = cfg['testing_epoch']
        self.Subset = cfg['test_subset']
        self.Finetuning = cfg['fine_tuning']
        self.Metrics = cfg['metrics']
        self.Desc = cfg['description']
        self.ReportIter= cfg['report_iter']
        self.Verbose = cfg['verbose']

        if cfg['data_source'] is None:
            self.DataSource = cfg['data_source']

        self.LoadType = cfg['load_type']

    def desc(self):
        desc = []
        for d in self.Desc:
            desc.append(d)

        # 额外添加一个last/best模型使用说明
        desc.append(f'{self.LoadType} used')

        # 额外添加一个测试数据子集说明
        if self.Subset != 'test':
            desc.append(f'{self.Subset} set used')

        # 额外添加任务信息
        task_info = f"{self.Task.Episode.k}-shot {self.Task.Episode.n}-way"
        desc.append(task_info)

        return desc
