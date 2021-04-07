from autorun.machine import ExecuteMachine

machine = ExecuteMachine(exe_bin='/opt/anaconda3/bin/python')

# machine.addRedoTrainTask(dataset='virushare-20',
#                          version=72,
#                          updated_configs={
#                              'task': {
#                                  'version': 87
#                              },
#                              'model': {
#                                  'model_name': 'SIMPLE'
#                              }
#                          })
machine.addTask('test',
                {
                    "task": {
                        "version": 87
                    },
                    'model_name': 'SIMPLE',
                    'load_type': 'best'
                })
machine.addTask('test',
                {
                    "task": {
                        "version": 87
                    },
                    'model_name': 'SIMPLE',
                    'load_type': 'last'
                })

# machine.addRedoTrainTask(dataset='virushare-20',
#                          version=73,
#                          updated_configs={
#                              'task': {
#                                  'version': 88
#                              },
#                              'model': {
#                                  'model_name': 'SIMPLE'
#                              }
#                          })
machine.addTask('test',
                {
                    "task": {
                        "version": 88
                    },
                    'model_name': 'SIMPLE',
                    'load_type': 'best'
                })
machine.addTask('test',
                {
                    "task": {
                        "version": 88
                    },
                    'model_name': 'SIMPLE',
                    'load_type': 'last'
                })

# machine.addRedoTrainTask(dataset='virushare-20',
#                          version=74,
#                          updated_configs={
#                              'task': {
#                                  'version': 89
#                              },
#                              'model': {
#                                  'model_name': 'SIMPLE'
#                              }
#                          })
machine.addTask('test',
                {
                    "task": {
                        "version": 89
                    },
                    'model_name': 'SIMPLE',
                    'load_type': 'best'
                })
machine.addTask('test',
                {
                    "task": {
                        "version": 89
                    },
                    'model_name': 'SIMPLE',
                    'load_type': 'last'
                })

# machine.addRedoTrainTask(dataset='virushare-20',
#                          version=77,
#                          updated_configs={
#                              'task': {
#                                  'version': 90
#                              },
#                              'model': {
#                                  'model_name': 'SIMPLE'
#                              }
#                          })
machine.addTask('test',
                {
                    "task": {
                        "version": 90
                    },
                    'model_name': 'SIMPLE',
                    'load_type': 'best'
                })
machine.addTask('test',
                {
                    "task": {
                        "version": 90
                    },
                    'model_name': 'SIMPLE',
                    'load_type': 'last'
                })

# machine.addRedoTrainTask(dataset='virushare-20',
#                          version=78,
#                          updated_configs={
#                              'task': {
#                                  'version': 91
#                              },
#                              'model': {
#                                  'model_name': 'SIMPLE'
#                              }
#                          })
machine.addTask('test',
                {
                    "task": {
                        "version": 91
                    },
                    'model_name': 'SIMPLE',
                    'load_type': 'best'
                })
machine.addTask('test',
                {
                    "task": {
                        "version": 91
                    },
                    'model_name': 'SIMPLE',
                    'load_type': 'last'
                })

# machine.addRedoTrainTask(dataset='virushare-20',
#                          version=79,
#                          updated_configs={
#                              'task': {
#                                  'version': 92
#                              },
#                              'model': {
#                                  'model_name': 'SIMPLE'
#                              }
#                          })
machine.addTask('test',
                {
                    "task": {
                        "version": 92
                    },
                    'model_name': 'SIMPLE',
                    'load_type': 'best'
                })
machine.addTask('test',
                {
                    "task": {
                        "version": 92
                    },
                    'model_name': 'SIMPLE',
                    'load_type': 'last'
                })

# machine.addRedoTrainTask(dataset='virushare-20',
#                          version=80,
#                          updated_configs={
#                              'task': {
#                                  'version': 93
#                              },
#                              'model': {
#                                  'model_name': 'SIMPLE'
#                              }
#                          })
machine.addTask('test',
                {
                    "task": {
                        "version": 93
                    },
                    'model_name': 'SIMPLE',
                    'load_type': 'best'
                })
machine.addTask('test',
                {
                    "task": {
                        "version": 93
                    },
                    'model_name': 'SIMPLE',
                    'load_type': 'last'
                })

# machine.addRedoTrainTask(dataset='virushare-20',
#                          version=81,
#                          updated_configs={
#                              'task': {
#                                  'version': 94
#                              },
#                              'model': {
#                                  'model_name': 'SIMPLE'
#                              }
#                          })
machine.addTask('test',
                {
                    "task": {
                        "version": 94
                    },
                    'model_name': 'SIMPLE',
                    'load_type': 'best'
                })
machine.addTask('test',
                {
                    "task": {
                        "version": 94
                    },
                    'model_name': 'SIMPLE',
                    'load_type': 'last'
                })



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
