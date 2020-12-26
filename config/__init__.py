import platform
import os

from .structs import *
from .structs import _loadJsonConfig
from .const import *


def _setCuda(task_config: TaskConfig):
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

_setCuda(task)

__all__ = ["env", "task", "train", "optimize", "params", "plot", "test"]

def printConfig():
    pass

