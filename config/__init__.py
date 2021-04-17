import platform
import warnings

from .structs import *
from .structs import _loadJsonConfig
from .const import *

cudaDevice = None

def _setCudaDevice(task_config: TaskConfig):
    global cudaDevice
    cudaDevice = task_config.DeviceId

    if task_config.DeviceId is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(task_config.DeviceId)


def _loadEnv():
    system_node = platform.node()
    env_cfg = _loadJsonConfig(file_name='env.json',
                              err_msg="没有合适的env config文件相对路径")
    return env_cfg['platform-node'][system_node]

env = EnvConfig(_loadEnv())
run_cfg = _loadJsonConfig(file_name="train.json",
                          err_msg="没有合适的run config文件相对路径")
task = TaskConfig(run_cfg)
train = TrainingConfig(run_cfg)
optimize = OptimizeConfig(run_cfg)
params = ParamsConfig(run_cfg)
plot = PlotConfig(run_cfg)

test_cfg = _loadJsonConfig(file_name="test.json",
                           err_msg="没有合适的test config文件相对路径")
test = TestConfig(test_cfg)

_setCudaDevice(task)

__all__ = ["env", "task", "train", "optimize", "params", "plot", "test"]


def printRunConfigSummary(task_config: TaskConfig=task, model_config: ParamsConfig=params):
    print("**************************************************")
    print(f"{model_config.ModelName} {task_config.Dataset} ver.{task_config.Version}")
    print(f"n:k:qk = {task_config.Episode.n}/{task_config.Episode.k}/{task_config.Episode.qk}")
    print(f"Cuda: {task_config.DeviceId}")
    print("**************************************************")


def reloadAllTestConfig(cfg_path):
    new_run_cfg = loadJson(cfg_path)

    # 测试时，只需要重加载模型参数和优化参数即可
    global optimize, params
    optimize = OptimizeConfig(new_run_cfg)
    params = ParamsConfig(new_run_cfg)

    # 加载训练时数据源作为测试时数据源
    _recoverTrainDataSource(new_run_cfg)
    # 加载训练时模型名称作为测试时模型名称
    _recoverTrainModelName(new_run_cfg)

    _setCudaDevice(test.Task)


def _recoverTrainDataSource(train_run_config):
    # 没有指定数据源时默认读取训练时数据源作为测试数据源
    if test.DataSource is None:
        test.DataSource = train_run_config['training']['data_source']


def _recoverTrainModelName(train_run_config):
    # 没有指定模型名称时，默认加载训练时模型名称作为测试模型
    if test.ModelName is None:
        test.ModelName = train_run_config['model']['model_name']


def reloadArbitraryConfig(new_cfg, reload_config_list):
    global task, train, optimize, params, plot
    # cfg_name_val_map = {
    #     'task': (lambda: task, lambda: TaskConfig(new_cfg)),
    #     'train': (lambda: train, lambda: TrainingConfig(new_cfg)),
    #     'optimize': (lambda: optimize, lambda: OptimizeConfig(new_cfg)),
    #     'params': (lambda: params, lambda: ParamsConfig(new_cfg)),
    #     'plot': (lambda: plot, lambda: PlotConfig(new_cfg))
    # }
    #
    # for cfg_name in config_name_list:
    #     given_val, giving_val = cfg_name_val_map[cfg_name]
    #     given_val = giving_val
    for cfg_name in reload_config_list:
        if cfg_name == 'task':
            task = TaskConfig(new_cfg)
        elif cfg_name == 'train':
            train = TrainingConfig(new_cfg)
        elif cfg_name == 'optimize':
            optimize = OptimizeConfig(new_cfg)
        elif cfg_name == 'params':
            params = ParamsConfig(new_cfg)
        elif cfg_name == 'plot':
            plot = PlotConfig(new_cfg)
        else:
            warnings.warn(f'[reloadListedConfig] Unrecognized config name: {cfg_name}')
