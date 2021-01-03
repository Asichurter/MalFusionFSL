import sys

sys.path.append('../')

import config
from utils.version import saveConfigFile
from utils.manager import *
from comp.dataset import FusionDataset
from builder import *
from utils.stat import statParamNumber
from utils.plot import plotLine

print('[train] Init managers...')
train_path_manager = PathManager(dataset=config.task.Dataset,
                                 subset="train",
                                 version=config.task.Version,
                                 model_name=config.params.ModelName)
val_path_manager = PathManager(dataset=config.task.Dataset,
                               subset='validate',
                               model_name=config.params.ModelName,
                               version=config.task.Version)

train_dataset = FusionDataset(train_path_manager.apiData(),
                              train_path_manager.imgData(),
                              train_path_manager.apiSeqLen(),
                              config.task.N)
val_dataset = FusionDataset(val_path_manager.apiData(),
                              val_path_manager.imgData(),
                              val_path_manager.apiSeqLen(),
                              config.task.N)

print('[train] building components...')
train_task = buildTask(train_dataset, config.task, config.params, config.optimize)
val_task = buildTask(val_dataset, config.task, config.params, config.optimize)

stat = buildStatManager(is_train=True,
                        path_manager=train_path_manager,
                        train_config=config.train)

plot = buildPlot(config.plot)
loss_func = buildLossFunc(optimize_config=config.optimize)
model = buildModel(path_manager=train_path_manager,
                   model_params=config.params,
                   loss_func=loss_func)
optimizer = buildOptimizer(named_parameters=model.named_parameters(),
                           optimize_params=config.optimize)
scheduler = buildScheduler(optimizer=optimizer,
                           optimize_params=config.optimize)

# 统计模型参数数量
statParamNumber(model)
# 保存运行配置文件
saveConfigFile(folder_path=train_path_manager.doc(), model_name=config.params.ModelName)

print("\n\n[train] Training starts!")
stat.begin()
for epoch in range(config.train.TrainEpoch):
    # print("# %d epoch"%epoch)

    model.train()

    loss_val = t.zeros((1,)).cuda()                 # loss需要cuda以保证优化时加速
    metrics = t.zeros((len(config.train.Metrics),)) # metric为了方便不使用cuda
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
    optimizer.step()
    scheduler.step()

    stat.recordTrain(metrics, loss_val.detach().item())

    # validate
    if (epoch+1) % config.train.ValCycle == 0:
        print("validate at %d epoch"%(epoch+1))
        model.eval()
        for val_i in range(config.train.ValEpisode):
            val_supports, val_querys = val_task.episode()
            val_outs = model(*val_supports, *val_querys, epoch=epoch)

            val_loss_val = val_outs['loss']
            if val_outs['logits'] is not None:
                val_metrics = val_task.metrics(val_outs['logits'],
                                            is_labels=False,
                                            metrics=config.train.Metrics)
            elif outs['predicts'] is not None:
                val_metrics = val_task.metrics(val_outs['predicts'],
                                              is_labels=True,
                                              metrics=config.train.Metrics)

            stat.recordVal(val_metrics, val_loss_val.detach().item(), model)

        train_metric, train_loss, val_metric, val_loss = stat.getRecentRecord(metric_idx=0)
        plot.update(title=config.train.Metrics[0], x_val=epoch+1, y_val=[[train_metric, val_metric]])
        plot.update(title='loss', x_val=epoch, y_val=[[train_loss, val_loss]])

stat.dumpStatHist()
plotLine(stat.getHistMetric(idx=0), ('train '+config.train.Metrics[0], 'val '+config.train.Metrics[0]),
         title=model.name()+' '+config.train.Metrics[0],
         gap=config.train.ValCycle,
         color_list=('blue', 'red'),
         style_list=('-','-'),
         save_path=train_path_manager.doc()+config.train.Metrics[0]+'.png')

plotLine(stat.getHistLoss(), ('train loss', 'val loss'),
         title=model.name()+' loss',
         gap=config.train.ValCycle,
         color_list=('blue', 'red'),
         style_list=('-','-'),
         save_path=train_path_manager.doc()+'loss.png')
