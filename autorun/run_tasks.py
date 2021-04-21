import os

from autorun.machine import ExecuteMachine
from utils.manager import PathManager

machine = ExecuteMachine(exe_bin='/opt/anaconda3/bin/python')

# tasks = {
#     336: {
#         'task|version': 336,
#         'model|model_name': 'ProtoNet',
#         'model|fusion|type': 'cat'
#     },
#     337: {
#         'task|version': 337,
#         'description|reproject': 'fusion之前使用了256的重投影',
#         'description|fusion': '使用image和sequence的add',
#         'model|model_name': 'ProtoNet',
#         'model|fusion|type': 'add',
#         'model|reproject|enabled': True,
#         'model|reproject|params|out_dim': 256
#     },
#     338: {
#         'task|version': 338,
#         'description|reproject': 'fusion之前使用了256的重投影',
#         'description|fusion': '使用image和sequence的prod',
#         'model|model_name': 'ProtoNet',
#         'model|fusion|type': 'prod',
#         'model|reproject|enabled': True,
#         'model|reproject|params|out_dim': 256
#     },
#     339: {
#         'task|version': 339,
#         'description|reproject': '无',
#         'description|fusion': '使用image和sequence的bilinear，输出维度为256,tanh激活，bn标准化',
#         'model|model_name': 'ProtoNet',
#         'model|reproject|enabled': False,
#         'model|fusion|type': 'bili',
#         'model|fusion|params|output_dim': 256
#     },
#     340: {
#         'task|version': 340,
#         'description|fusion': '使用image和sequence的hmd_bili，隐藏维度512，输出维度为256,tanh激活，bn标准化',
#         'model|model_name': 'ProtoNet',
#         'model|fusion|type': 'hdm_bili',
#         'model|fusion|params|output_dim': 256,
#         'model|fusion|params|hidden_dim': 512
#     },
#     341: {
#         'task|version': 341,
#         'description|img_embed': '使用经过了stride的conv-n，不经过global_pooling，输出patch',
#         'description|fusion': '使用image和sequence的seq_attend_img_att_only,隐藏维度为512，不使用注意力缩放因子',
#         'model|model_name': 'ProtoNet',
#         'model|fusion|type': 'seq_attend_img_att_only',
#         'model|fusion|params|hidden_dim': 512,
#         'model|conv_backbone|params|conv-n|global_pooling': False,
#         'model|conv_backbone|params|conv-n|out_type': 'patch',
#     },
#     342: {
#         'task|version': 342,
#         'description|img_embed': '使用经过了stride的conv-n，不经过global_pooling，输出patch',
#         'description|fusion': '使用image和sequence的seq_attend_img_cat,隐藏维度为512，不使用注意力缩放因子',
#         'model|model_name': 'ProtoNet',
#         'model|fusion|type': 'seq_attend_img_cat',
#         'model|fusion|params|hidden_dim': 512,
#         'model|conv_backbone|params|conv-n|global_pooling': False,
#         'model|conv_backbone|params|conv-n|out_type': 'patch',
#     },
#     343: {
#         'task|version': 343,
#         'description|img_embed': '使用经过了stride的conv-n，使用global_pooling',
#         'description|fusion': '使用image和sequence的dnn_cat,三层，同126配置',
#         'model|model_name': 'ProtoNet',
#         'model|fusion|type': 'dnn_cat',
#         'model|conv_backbone|params|conv-n|global_pooling': True,
#     },
#     344: {
#         'task|version': 344,
#         'description|img_embed': '使用经过了stride的conv-n，使用global_pooling',
#         'description|fusion': '使用image和sequence的dnn_cat_ret_cat,三层，同126配置',
#         'model|model_name': 'ProtoNet',
#         'model|fusion|type': 'dnn_cat_ret_cat'
#     },
# }

# for ver, conf in tasks.items():
#     machine.addTask('train', flatten_update_config=conf)
#     machine.addTask('test', flatten_update_config={
#         'task|version': ver,
#         'load_type': 'best',
#     })
#     machine.addTask('test', flatten_update_config={
#         'task|version': ver,
#         'load_type': 'last',
#     })

s_ver, e_ver = 318, 326
for ver in range(s_ver, e_ver + 1):
    pm = PathManager(dataset='virushare-20',
                     version=ver)
    if os.path.exists(pm.doc()+'test_result.json'):
        os.remove(pm.doc()+'test_result.json')
    machine.addTask('test', flatten_update_config={
        'task|version': ver,
        'load_type': 'best',
    })
    machine.addTask('test', flatten_update_config={
        'task|version': ver,
        'load_type': 'last',
    })



# machine.addTask('train',
#                 {
#                     'task': {
#                         'version': 214
#                     },
#                     "description": [
#                         "序列长度为300",
#                         "使用sgd优化",
#                         "使用1层BiLSTM编码,使用1dCNN解码",
#                         "使用经过了stride的conv-n，使用global_pooling",
#                         "使用image和sequence的dnn_cat_ret,3层，使用与94相同的维度，最后使用relu激活",
#                         "初始学习率为1e-3",
#                         "辅助损失系数aux_loss_factor为0.5"
#                     ],
#                     'model': {
#                         'more': {
#                             'axu_loss_factor': 0.5
#                         }
#                     }
#                 })

# machine.addTask('train',
#                 {
#                     'task': {
#                         'version': 1000
#                     },
#                     'training': {
#                         'epoch': 50000,
#                         'device_id': 3,
#                         'clip_grad_norm': None
#                     },
#                     'description': [
#                         "序列长度为300",
#                         "使用sgd优化",
#                         "使用经过了stride的conv-n，使用global_pooling，四层，通道数增加到512",
#                         "使用image",
#                         "初始学习率为1e-3"
#                     ],
#                     'model': {
#                         'conv_backbone': {
#                             'params': {
#                                 'conv-n': {
#                                     'channels': [1, 64, 128, 256, 512],
#                                     'kernel_sizes': [3, 3, 3, 3],
#                                     'padding_sizes': [1, 1, 1, 1],
#                                     'strides': [2, 1, 1, 1],
#                                     'nonlinears': [True, True, True, False]
#                                 }
#                             }
#                         }
#                     }
#                 })
# machine.addTask('test',
#                 {
#                     "task": {
#                         "version": 1000
#                     }
#                 })
#
# machine.addTask('train',
#                 {
#                     'task': {
#                         'version': 1001
#                     },
#                     'training': {
#                         'epoch': 50000,
#                         'device_id': 3,
#                         'clip_grad_norm': None
#                     },
#                     'description': [
#                         "序列长度为300",
#                         "使用sgd优化",
#                         "使用没有stride的conv-n，使用global_pooling，四层，通道数增加到512",
#                         "使用image",
#                         "初始学习率为1e-3"
#                     ],
#                     'model': {
#                         'conv_backbone': {
#                             'params': {
#                                 'conv-n': {
#                                     'channels': [1, 64, 128, 256, 512],
#                                     'kernel_sizes': [3, 3, 3, 3],
#                                     'padding_sizes': [1, 1, 1, 1],
#                                     'strides': [1, 1, 1, 1],
#                                     'nonlinears': [True, True, True, False]
#                                 }
#                             }
#                         }
#                     }
#                 })
# machine.addTask('test',
#                 {
#                     "task": {
#                         "version": 1001
#                     }
#                 })
#
# machine.addTask('train',
#                 {
#                     'task': {
#                         'version': 1002
#                     },
#                     'training': {
#                         'epoch': 50000,
#                         'device_id': 3,
#                         'clip_grad_norm': None
#                     },
#                     'description': [
#                         "序列长度为300",
#                         "使用sgd优化",
#                         "使用没有stride的conv-n，使用global_pooling，五层，通道数增加到1024",
#                         "使用image",
#                         "初始学习率为1e-3"
#                     ],
#                     'model': {
#                         'conv_backbone': {
#                             'params': {
#                                 'conv-n': {
#                                     'channels': [1, 64, 128, 256, 512, 1024],
#                                     'kernel_sizes': [3, 3, 3, 3, 3],
#                                     'padding_sizes': [1, 1, 1, 1, 1],
#                                     'strides': [1, 1, 1, 1, 1],
#                                     'nonlinears': [True, True, True, True, False]
#                                 }
#                             }
#                         }
#                     }
#                 })
# machine.addTask('test',
#                 {
#                     "task": {
#                         "version": 1002
#                     }
#                 })
#
# machine.addTask('train',
#                 {
#                     'task': {
#                         'version': 1003
#                     },
#                     'training': {
#                         'epoch': 60000,
#                         'device_id': 3,
#                         'clip_grad_norm': None
#                     },
#                     'description': [
#                         "序列长度为300",
#                         "使用sgd优化",
#                         "使用没有stride的conv-n，使用global_pooling，六层，通道数增加到2048",
#                         "使用image",
#                         "初始学习率为1e-3"
#                     ],
#                     'model': {
#                         'conv_backbone': {
#                             'params': {
#                                 'conv-n': {
#                                     'channels': [1, 64, 128, 256, 512, 1024, 2048],
#                                     'kernel_sizes': [3, 3, 3, 3, 3, 3],
#                                     'padding_sizes': [1, 1, 1, 1, 1, 1],
#                                     'strides': [1, 1, 1, 1, 1, 1],
#                                     'nonlinears': [True, True, True, True, True, False]
#                                 }
#                             }
#                         }
#                     }
#                 })
# machine.addTask('test',
#                 {
#                     "task": {
#                         "version": 1003
#                     }
#                 })

machine.execute()
