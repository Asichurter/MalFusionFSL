import shutil
import random
import os
from tqdm import tqdm

from utils.magic import magicSeed
from utils.os import rmAll
from utils.file import loadJson, dumpJson
from utils.manager import PathManager

##########################################################
# 本文件是为了将数据集中的类进行随机抽样移动，可以用于数据集分割
##########################################################
def _splitDatas(src, dest, ratio, mode='x', is_dir=False):
    '''
    将生成的样本按比例随机抽样分割，并且移动到指定文件夹下，用于训练集和验证集的制作
    src:源文件夹
    dest:目标文件夹
    ratio:分割比例或者最大数量
    '''
    assert mode in ['c', 'x'], '选择的模式错误，只能复制c或者剪切x'

    all_items = os.listdir(src)

    if ratio < 0:                   # 比例flag为负，代表选中所有
        size = len(all_items)
    elif 1 > ratio > 0:
        size = int(len(all_items) * ratio)
    else:
        size = ratio

    assert len(all_items) >= size, '分割时，总数量没有要求的数量大！'

    random.seed(magicSeed())
    samples_names = random.sample(all_items, size)
    num = 0
    for item in tqdm(all_items):
        if item in samples_names:
            num += 1
            path = src + item
            if mode == 'x':
                shutil.move(path, dest)
            else:
                if is_dir:
                    shutil.copytree(src=path, dst=dest+item)
                else:
                    shutil.copy(src=path, dst=dest)


def _redoSplit(src_path, dst_path, based_path):
    for folder in os.listdir(based_path):
        shutil.move(src_path+folder,
                    dst_path+folder)


def splitDataset(dataset_path, validate_ratio=20, test_ratio=20):
    print("[SplitDataset] Clearing...")
    rmAll(dataset_path+'train/api/')
    rmAll(dataset_path+'train/img/')
    rmAll(dataset_path+'validate/api/')
    rmAll(dataset_path+'validate/img/')
    rmAll(dataset_path+'test/api/')
    rmAll(dataset_path+'test/img/')

    print("[SplitDataset] Copy...")
    _splitDatas(dataset_path+'all/api/', dataset_path+'train/api/', mode='c', ratio=-1, is_dir=True)
    _splitDatas(dataset_path+'all/img/', dataset_path+'train/img/', mode='c', ratio=-1, is_dir=True)

    print("[SplitDataset] Splitting validate set...")
    _splitDatas(dataset_path+'train/api/', dataset_path+'validate/api/', mode='x', ratio=validate_ratio, is_dir=True)
    _redoSplit(dataset_path+'train/img/', dataset_path+'validate/img/', dataset_path+'validate/api/')

    print("[SplitDataset] Splitting test set...")
    _splitDatas(dataset_path+'train/api/', dataset_path+'test/api/', mode='x', ratio=test_ratio, is_dir=True)
    _redoSplit(dataset_path + 'train/img/', dataset_path + 'test/img/', dataset_path + 'test/api/')


##########################################################
# 本函数用于将训练集，验证集和测试集的数据集分割详情保存到JSON文件
# 以便复现出实验结果
##########################################################
def dumpDatasetSplitStruct(base_path, dump_path, desc: list):
    dump = {"desc": desc}
    for split in ['train', 'validate', 'test']:
        print(f"[dumpDatasetSplitStruct] {split}")
        folders = []

        for folder in os.listdir(base_path+split+'/api'):   # 以api文件夹的为标准
            folders.append(folder)

        dump[split] = folders

    dumpJson(dump, dump_path)
    print('-- Done --')

##########################################################
# 本函数用于删除原有数据集train，validate和test文件夹中的数据
##########################################################
def deleteDatasetSplit(dataset_base):
    for typ in ['train','validate','test']:
        print(f"[deleteDatasetSplit] {typ}")
        rmAll(dataset_base+typ+'/api')
        rmAll(dataset_base+typ+'/img')

##########################################################
# 本函数用于将训练集，验证集和测试集的数据集分割详情根据保存的JSON文件
# 还原到数据集分隔文件夹中。
##########################################################
def revertDatasetSplit(dataset, dump_path):
    man = PathManager(dataset)
    split_dump = loadJson(dump_path)

    deleteDatasetSplit(man.datasetBase())

    for typ in ['train', 'validate', 'test']:
        print(f"[revertDatasetSplit] {typ}")
        for folder in split_dump[typ]:
            shutil.copytree(src=man.datasetBase()+'all/api/'+folder+'/',
                            dst=man.datasetBase()+typ+'/api/'+folder+'/')
            shutil.copytree(src=man.datasetBase()+'all/img/'+folder+'/',
                            dst=man.datasetBase()+typ+'/img/'+folder+'/')

    print('-- Done --')