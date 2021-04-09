import os
from pprint import pprint

from utils.file import loadJson, dumpJson
from utils.manager import PathManager
from utils.os import joinPath


def _extractSummaryInformation(dataset, version, train_config, test_config):
    packed_info = {
        'dataset': dataset,
        'version': version
    }

    packed_info['model_name'] = train_config['model']['model_name']
    packed_info['desc'] = train_config['description']
    packed_info['test_result'] = test_config['test_result']

    return packed_info


def makeResultSummaryByVerRange(dataset,
                                version_range=[0,1],
                                dump_path=None):
    # 左闭右开区间变为闭区间
    version_range[1] += 1
    summaries = []
    for version in range(*version_range):
        pm = PathManager(dataset, version=version)
        try:
            train_config = loadJson(pm.doc()+'train.json')
            test_config = loadJson(pm.doc()+'test_result.json')
        except FileNotFoundError:
            continue

        extracted_info = _extractSummaryInformation(dataset, version, train_config, test_config)
        summaries.append(extracted_info)

    pprint(summaries, indent=4)

    if dump_path is not None:
        dumpJson(summaries, dump_path)


def _checkRecursiveCond(cur_cfg, cur_cond):
    for k,v in cur_cond.items():
        # 条件中的key不存在于config中，直接返回false
        if k not in cur_cfg:
            # print(f"key '{k}' not in config: {cur_cfg}")
            return False
        # value的类型不同，直接返回false
        if type(cur_cfg[k]) != type(v):
            # print(f"value type '{type(v)}' not equal to config value type: {type(cur_cfg[k])}")
            return False
        # 如果value是dict类型，需要递归
        if type(v) == dict:
            res = _checkRecursiveCond(cur_cfg[k], v)
            if not res:
                return False
        # 否则直接比值
        else:
            res = (cur_cfg[k] == v)
            if not res:
                # print(f"value '{v}' not equal to config value: {cur_cfg[k]}")
                return False
    return True


def makeResultByTrainConfigCond(dataset,
                                train_config_cond={},
                                dump_path=None):
    summaries = []
    pm = PathManager(dataset)
    version_folders = sorted(os.listdir(pm.docBase()))
    for folder in version_folders:
        try:
            train_config = loadJson(joinPath(pm.docBase(), folder, 'train.json'))
            test_config = loadJson(joinPath(pm.docBase(), folder, 'test_result.json'))
        except FileNotFoundError:
            continue

        if not _checkRecursiveCond(train_config, train_config_cond):
            continue

        extracted_info = _extractSummaryInformation(dataset, folder, train_config, test_config)
        summaries.append(extracted_info)

    pprint(summaries, indent=4)

    if dump_path is not None:
        dumpJson(summaries, dump_path)
