from autorun.machine import ExecuteMachine

machine = ExecuteMachine(exe_bin='/opt/anaconda3/bin/python')

# machine.addTask('train',
#                 flatten_update_config={
#                     'task|version': 133,
#                     'description|other-1': '参数基本同125，在fusion输出前添加了layer_norm',
#                     # 'description|fusion': 'test fusion',
#                     'model|fusion|params|dnn_norm_type': 'ln',
#                     # 'model|more|aux_loss_seq_factor': 0.5
#                 })
# machine.addTask('test',
#                 {
#                     "task": {
#                         "version": 133
#                     },
#                     'load_type': 'best'
#                 })
# machine.addTask('test',
#                 {
#                     "task": {
#                         "version": 133
#                     },
#                     'load_type': 'last'
#                 })


machine.addTask('train',
                flatten_update_config={
                    'task|version': 134,
                    'description|other-1': '',
                    # 'description|fusion': 'test fusion',
                    'model|fusion|params|dnn_norm_type': 'none',
                    'model|more|aux_loss_seq_factor': 0.5
                })
machine.addTask('test',
                {
                    "task": {
                        "version": 134
                    },
                    'load_type': 'best'
                })
machine.addTask('test',
                {
                    "task": {
                        "version": 134
                    },
                    'load_type': 'last'
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
