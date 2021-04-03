import sys
import torch
from tqdm import tqdm

sys.path.append('../')

import config
from utils.version import saveConfigFile, saveRunVersionConfig
from utils.manager import *
from comp.dataset import FusedDataset
from builder import *
from utils.stat import statParamNumber
from utils.plot import plotLine

if config.train.Verbose:
    print('[train] Init managers...')
train_path_manager = PathManager(dataset=config.task.Dataset,
                                 subset="train",
                                 version=config.task.Version,
                                 model_name=config.params.ModelName)
val_path_manager = PathManager(dataset=config.task.Dataset,
                               subset='validate',
                               model_name=config.params.ModelName,
                               version=config.task.Version)

train_dataset = FusedDataset(train_path_manager.apiData(),
                             train_path_manager.imgData(),
                             train_path_manager.apiSeqLen(),
                             config.task.N,
                             config.train.DataSource)
val_dataset = FusedDataset(val_path_manager.apiData(),
                           val_path_manager.imgData(),
                           val_path_manager.apiSeqLen(),
                           config.task.N,
                           config.train.DataSource)

if config.train.Verbose:
    print('[train] building components...')
train_task = buildTask(train_dataset, config.task, config.params, config.optimize, "Train")
val_task = buildTask(val_dataset, config.task, config.params, config.optimize, "Validate")

stat = buildStatManager(is_train=True,
                        task_config=config.task,
                        path_manager=train_path_manager,
                        train_config=config.train)

plot = buildPlot(config.plot)
loss_func = buildLossFunc(optimize_config=config.optimize)
model = buildModel(path_manager=train_path_manager,
                   task_config=config.task,
                   model_params=config.params,
                   loss_func=loss_func,
                   data_source=config.train.DataSource)
optimizer = buildOptimizer(named_parameters=model.named_parameters(),
                           optimize_params=config.optimize)
scheduler = buildScheduler(optimizer=optimizer,
                           optimize_params=config.optimize)

# 统计模型参数数量
if config.train.Verbose:
    statParamNumber(model)
# 保存运行配置文件
saveConfigFile(config.params,
               folder_path=train_path_manager.doc(),
               dataset_base=train_path_manager.datasetBase(),
               model_name=config.params.ModelName)
# 保存训练信息
saveRunVersionConfig(config.task,
                     config.params,
                     config.train.Desc)

# model.load_state_dict(torch.load('F:/FSL_mal_data/datasets/LargePE-Per35/models/ProtoNet_v.13_latest'))

if config.train.Verbose:
    print("\n\n[train] Training starts!")
stat.begin()

if config.train.Verbose:
    epoch_range = range(config.train.TrainEpoch)
else:
    epoch_range = tqdm(range(config.train.TrainEpoch))

for epoch in epoch_range:
    # print("# %d epoch"%epoch)

    model.train_state()
    # 1.17修复：梯度没有归0导致优化进行时梯度累计
    model.zero_grad()

    loss_val = t.zeros((1,)).cuda()  # loss需要cuda以保证优化时加速
    metrics = np.zeros((len(config.train.Metrics),))
    for i in range(config.optimize.TaskBatch):
        supports, querys = train_task.episode()
        outs = model(*supports, *querys, epoch=epoch)

        loss_val += outs['loss']
        if outs['logits'] is not None:
            metrics += train_task.metrics(outs['logits'],
                                          is_labels=False,
                                          metrics=config.train.Metrics)
        elif outs['predicts'] is not None:
            metrics += train_task.metrics(outs['predicts'],
                                          is_labels=True,
                                          metrics=config.train.Metrics)

    loss_val /= config.optimize.TaskBatch
    metrics /= config.optimize.TaskBatch

    loss_val.backward()
    # 裁剪梯度
    if config.train.ClipGradNorm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.ClipGradNorm)

    optimizer.step()
    scheduler.step()

    stat.recordTrain(metrics, loss_val.detach().item())

    # validate
    if (epoch + 1) % config.train.ValCycle == 0:
        if config.train.Verbose:
            print("validate at %d epoch" % (epoch + 1))
        model.validate_state()
        for val_i in range(config.train.ValEpisode):
            supports, querys = val_task.episode()
            # t.cuda.empty_cache()
            outs = model.test(*supports, *querys, epoch=epoch)

            loss_val = outs['loss']
            if outs['logits'] is not None:
                metrics = val_task.metrics(outs['logits'],
                                           is_labels=False,
                                           metrics=config.train.Metrics)
            elif outs['predicts'] is not None:
                metrics = val_task.metrics(outs['predicts'],
                                           is_labels=True,
                                           metrics=config.train.Metrics)
            else:
                raise RuntimeError("[Train] Either logits or predicts must be returned by the model's forward")

            stat.recordVal(metrics, loss_val.detach().item(), model)

        if config.train.Verbose:
            train_metric, train_loss, val_metric, val_loss = stat.getRecentRecord(metric_idx=0)
            plot.update(title=config.train.Metrics[0], x_val=epoch+1, y_val=[[train_metric, val_metric]],
                        update={'flag': True, 'val': None if (epoch+1) // config.train.ValCycle <= 1 else 'append'})
            plot.update(title='loss', x_val=epoch+1, y_val=[[train_loss, val_loss]],
                        update={'flag': True, 'val': None if (epoch+1) // config.train.ValCycle <= 1 else 'append'})

stat.dumpStatHist()
plotLine(stat.getHistMetric(idx=0), ('train ' + config.train.Metrics[0], 'val ' + config.train.Metrics[0]),
         title=model.name() + ' ' + config.train.Metrics[0],
         gap=config.train.ValCycle,
         color_list=('blue', 'red'),
         style_list=('-', '-'),
         save_path=train_path_manager.doc() + config.train.Metrics[0] + '.png')

plotLine(stat.getHistLoss(), ('train loss', 'val loss'),
         title=model.name() + ' loss',
         gap=config.train.ValCycle,
         color_list=('blue', 'red'),
         style_list=('-', '-'),
         save_path=train_path_manager.doc() + 'loss.png')
