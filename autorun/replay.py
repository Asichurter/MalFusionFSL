##################################################################
# 根据训练时记录的数据，重新在可视化工具（Visdom...）中进行数据可视化复盘
##################################################################

import numpy as np
from pprint import pprint

import config
from builder import buildPlot
from utils.file import loadJson
from utils.manager import PathManager

dataset = 'virushare-20'
version = 26
model = 'ProtoNet'
report_iter = 100
val_episode = 50

pm = PathManager(dataset, version=version, model_name=model)

train_stat_data = loadJson(pm.doc()+'train_stat.json')
train_config = loadJson(pm.doc()+'train.json')
config.reloadArbitraryConfig(train_config, reload_config_list=['plot'])
config.plot.Enabled = True
vis = buildPlot(config.plot)

train_metric = np.array(train_stat_data['train']['metrics']).reshape((-1, report_iter))     # 此处假设metric_num=1
train_loss = np.array(train_stat_data['train']['loss']).reshape((-1, report_iter))
# validate数据的总数与validate_episode有关
val_metric = np.array(train_stat_data['validate']['metrics']).reshape((-1, val_episode))     # 此处假设metric_num=1
val_loss = np.array(train_stat_data['validate']['loss']).reshape((-1, val_episode))

train_metric = train_metric.mean(axis=1)
train_loss = train_loss.mean(axis=1)
val_metric = val_metric.mean(axis=1)
val_loss = val_loss.mean(axis=1)

for i, (t_m, t_l, v_m, v_l) in enumerate(zip(train_metric, train_loss, val_metric, val_loss)):
    vis.update('acc', (i+1)*report_iter, [[t_m, v_m]],
               update={'flag': True, 'val': None if i==0 else 'append'})
    vis.update('loss', (i+1)*report_iter, [[t_l, v_l]],
               update={'flag': True, 'val': None if i==0 else 'append'})
