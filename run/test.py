import sys
from tqdm import tqdm

sys.path.append('../')

import config

from utils.version import saveConfigFile
from utils.manager import *
from comp.dataset import FusedDataset
from builder import *
from utils.stat import statParamNumber
from utils.plot import plotLine

if config.test.Verbose:
    print('[test] Init managers...')
test_path_manager = PathManager(dataset=config.test.Task.Dataset,
                                subset=config.test.Subset,
                                version=config.test.Task.Version,
                                model_name=config.test.ModelName)
# 重新加载训练时的params和optimize参数
config.reloadAllTestConfig(test_path_manager.doc() + 'train.json')
# 由于初始化manager时尚未知道model_name，需要在此处重新设置一次加载到的model_name
test_path_manager.setModelName(config.test.ModelName)

test_dataset = FusedDataset(test_path_manager.apiData(),
                            test_path_manager.imgData(),
                            test_path_manager.apiSeqLen(),
                            config.test.Task.N,
                            config.test.DataSource)

if config.test.Verbose:
    print('[test] building components...')
test_task = buildTask(test_dataset, config.test.Epoch,
                      config.test.Task, config.params, config.optimize, "Test")

stat = buildStatManager(is_train=False,
                        path_manager=test_path_manager,
                        test_config=config.test)

loss_func = buildLossFunc(optimize_config=config.optimize)
model = buildModel(path_manager=test_path_manager,
                   model_params=config.params,
                   task_config=config.test.Task,
                   loss_func=loss_func,
                   data_source=config.test.DataSource)

state_dict = t.load(test_path_manager.model(load_type=config.test.LoadType))
model.load_state_dict(state_dict)

# 统计模型参数数量
if config.test.Verbose:
    statParamNumber(model)
    print("\n\n[test] Testing starts!")

stat.begin()

if config.test.Verbose:
    epoch_range = range(config.test.Epoch)
else:
    epoch_range = tqdm(range(config.test.Epoch))

for epoch in epoch_range:
    # print("# %d epoch"%epoch)

    model.test_state()

    supports, querys = test_task.episode()
    outs = model.test(*supports, *querys, epoch=epoch)

    loss_val = outs['loss']
    if outs['logits'] is not None:
        metrics = test_task.metrics(outs['logits'],
                                    is_labels=False,
                                    metrics=config.test.Metrics)
    elif outs['predicts'] is not None:
        metrics = test_task.metrics(outs['predicts'],
                                    is_labels=True,
                                    metrics=config.test.Metrics)
    else:
        raise ValueError("[test] 'logits' or 'predicts' must be either given in returned dict of model")

    stat.recordTest(metrics, loss_val.detach().item())

stat.dump(path=test_path_manager.doc()+'test_result.json',
          desc=config.test.desc())
