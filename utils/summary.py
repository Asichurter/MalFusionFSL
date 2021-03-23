from utils.file import loadJson, dumpJson
from utils.manager import PathManager
from pprint import pprint


def makeResultSummary(dataset,
                      version_range=[0,1],
                      dump_path=None):
    # 左闭右开区间变为闭区间
    version_range[1] += 1
    summaries = []
    for version in range(*version_range):
        ver_sum = {
            'dataset': dataset,
            'version': version,
        }
        pm = PathManager(dataset, version=version)
        try:
            train_config = loadJson(pm.doc()+'train.json')
            test_results = loadJson(pm.doc()+'test_result.json')
        except FileNotFoundError:
            continue
        ver_sum['model_name'] = train_config['model']['model_name']
        ver_sum['desc'] = train_config['description']
        ver_sum['test_result'] = test_results['test_result']

        summaries.append(ver_sum)

    pprint(summaries)

    if dump_path is not None:
        dumpJson(summaries, dump_path)
