import sys

sys.path.append('../')

import config
from utils.version import saveConfigFile
from utils.manager import *
from comp.dataset import FusionDataset
from builder import *
from utils.stat import statParamNumber

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
model = buildModel(path_manager=train_path_manager,
                   model_params=config.params)
optimizer = buildOptimizer(named_parameters=model.named_parameters(),
                           optimize_params=config.optimize)
scheduler = buildScheduler(optimizer=optimizer,
                           optimize_params=config.optimize)
loss_func = buildLossFunc(optimize_config=config.optimize)

statParamNumber(model)
stat.begin()

# 保存运行配置文件
saveConfigFile(folder_path=train_path_manager,
               model_name=config.params.ModelName)

print("[train] Training starts !")
for epoch in range(config.train.TrainEpoch):

    model.train()

    loss_val = t.zeros((1,)).cuda()                 # loss需要cuda以保证优化时加速
    metrics = t.zeros((len(config.train.Metrics),)) # metric为了方便不使用cuda
    for i in range(config.optimize.TaskBatch):
        supports, querys = train_task.episode()
        outs = model(*supports, *querys)

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

    # TODO: validate

