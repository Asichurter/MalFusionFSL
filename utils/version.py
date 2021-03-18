import shutil
import os
import time

import config
from preprocessing.dataset.split import dumpDatasetSplitStruct
from utils.file import loadJson, dumpJson


def saveConfigFile(model_params, folder_path, model_name, dataset_base):
    # 若目前的version已经存在，则删除之
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.mkdir(folder_path)
    # 复制运行配置文件
    shutil.copy('../config/train.json', folder_path + 'train.json')
    # 保存数据集分割信息
    dumpDatasetSplitStruct(dataset_base,
                           folder_path+'dataset_struct.json',
                           desc=[])
    # TODO: 保存模型结构以保证可复现性
    # 复制模型代码文件
    # shutil.copy(f'../models/{model_name}.py', folder_path + f'{model_name}.py')

    # 创建一个指示model的空文件
    os.mkdir(folder_path+model_params.ModelName)


###########################################################
# 将运行时的配置,如版本,数据集,模型,模型参数,运行时间等保存到文件中
###########################################################
def saveRunVersionConfig(task_params: config.TaskConfig,
                         model_params: config.ParamsConfig,
                         decs=None,
                         ver_cfg_path='../config/version.json'):
    task_config = {
        'k': task_params.Episode.k,
        'n': task_params.Episode.n,
        'qk': task_params.Episode.qk,
        'decs': decs
    }

    cfg_pack = {'__version': task_params.Version,
                '_model': model_params.ModelName,
                '_dataset': task_params.Dataset,
                'config': task_config,
                '_time': time.asctime()}
    try:
        ver_cfg = loadJson(ver_cfg_path)
    except FileNotFoundError:
        ver_cfg = {}
    ver_cfg[str(task_params.Version)] = cfg_pack

    dumpJson(ver_cfg, ver_cfg_path, sort=True)