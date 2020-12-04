import platform
import os

from utils.file import loadJson

# 所有可能的启动路径的config文件相对路径
_REL_CFG_PATHS = [
    "../config/env.json",       # 启动位置位于某个包下
    "config/env.json"           # 启动位置位于项目根目录，例如python console
]

class EnvConfig:
    def __init__(self, cfg):
        self.DatasetBasePath = cfg['DatasetBasePath']
        self.ReportPath = cfg['ReportPath']

def _loadEnv():
    system_node = platform.node()
    for cfg_path in _REL_CFG_PATHS:
        try:
            env_cfg = loadJson(cfg_path)
        except:
            continue
        return env_cfg['platform-node'][system_node]
    raise RuntimeError("没有合适的env config文件相对路径")

env = EnvConfig(_loadEnv())

__all__ = [env]

