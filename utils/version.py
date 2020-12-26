import shutil
import os

def saveConfigFile(folder_path, model_name):
    # 若目前的version已经存在，则删除之
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.mkdir(folder_path)
    # 复制运行配置文件
    shutil.copy('../config/train.json', folder_path + 'train.json')
    # TODO: 保存模型结构以保证可复现性
    # 复制模型代码文件
    # shutil.copy(f'../models/{model_name}.py', folder_path + f'{model_name}.py')