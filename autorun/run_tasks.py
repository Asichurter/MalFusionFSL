from autorun.machine import ExecuteMachine

machine = ExecuteMachine(exe_bin='/opt/anaconda3/bin/python')
machine.addTask('train',
                {
                    'task': {
                        'version': 59,
                        'preload_state_dict_versions': []
                    },
                    'description': [
                        "使用300维度的GloVe初始化",
                        "序列长度为300",
                        "提前终止的标准为loss",
                        "使用sgd优化",
                        "使用1层BiLSTM编码,使用1dCNN解码",
                        "使用没有经过stride的14×14的patch的conv-n",
                        "使用image和sequence的序列到图像patch注意力对齐,只返回att",
                        "注意力中间添加0.2的dropout",
                        "不！使用image和sequence的预训练参数preload",
                        "初始学习率为1e-3",
                        "不使用重投影。FCProject: 无非线性激活,dim=64"
                    ],
                    'model': {
                        'fusion': {
                            'type': 'seq_attend_img_att_only'
                        },
                        'conv_backbone': {
                            'params': {
                                'conv-n': {
                                    'strides': [1,1,1,1]
                                }
                            }
                        }
                    }
                })
machine.addTask('train',
                {
                    'task': {
                        'version': 60,
                        'preload_state_dict_versions': [15,16]
                    },
                    'description': [
                        "使用300维度的GloVe初始化",
                        "序列长度为300",
                        "提前终止的标准为loss",
                        "使用sgd优化",
                        "使用1层BiLSTM编码,使用1dCNN解码",
                        "使用没有经过stride的14×14的patch的conv-n",
                        "使用image和sequence的序列到图像patch注意力对齐,只返回att",
                        "注意力中间添加0.2的dropout",
                        "使用image和sequence的预训练参数preload",
                        "初始学习率为1e-3",
                        "不使用重投影。FCProject: 无非线性激活,dim=64"
                    ],
                    'model': {
                        'fusion': {
                            'type': 'seq_attend_img_att_only'
                        },
                        'conv_backbone': {
                            'params': {
                                'conv-n': {
                                    'strides': [1,1,1,1]
                                }
                            }
                        }
                    }
                })
machine.addTask('train',
                {
                    'task': {
                        'version': 61,
                        'preload_state_dict_versions': []
                    },
                    'description': [
                        "使用300维度的GloVe初始化",
                        "序列长度为300",
                        "提前终止的标准为loss",
                        "使用sgd优化",
                        "使用1层BiLSTM编码,使用1dCNN解码",
                        "使用经过了stride的7×7的patch的conv-n",
                        "使用image和sequence的序列到图像patch注意力对齐,只返回att",
                        "注意力中间添加0.2的dropout",
                        "不！使用image和sequence的预训练参数preload",
                        "初始学习率为1e-3",
                        "不使用重投影。FCProject: 无非线性激活,dim=64"
                    ],
                    'model': {
                        'fusion': {
                            'type': 'seq_attend_img_att_only'
                        },
                        'conv_backbone': {
                            'params': {
                                'conv-n': {
                                    'strides': [2,1,1,1]
                                }
                            }
                        }
                    }
                })
machine.addTask('train',
                {
                    'task': {
                        'version': 62,
                        'preload_state_dict_versions': [14,15]
                    },
                    'description': [
                        "使用300维度的GloVe初始化",
                        "序列长度为300",
                        "提前终止的标准为loss",
                        "使用sgd优化",
                        "使用1层BiLSTM编码,使用1dCNN解码",
                        "使用经过了stride的7×7的patch的conv-n",
                        "使用image和sequence的序列到图像patch注意力对齐,只返回att",
                        "注意力中间添加0.2的dropout",
                        "使用image和sequence的预训练参数preload",
                        "初始学习率为1e-3",
                        "不使用重投影。FCProject: 无非线性激活,dim=64"
                    ],
                    'model': {
                        'fusion': {
                            'type': 'seq_attend_img_att_only'
                        },
                        'conv_backbone': {
                            'params': {
                                'conv-n': {
                                    'strides': [2,1,1,1]
                                }
                            }
                        }
                    }
                })
machine.addTask('train',
                {
                    'task': {
                        'version': 63,
                        'preload_state_dict_versions': []
                    },
                    'description': [
                        "使用300维度的GloVe初始化",
                        "序列长度为300",
                        "提前终止的标准为loss",
                        "使用sgd优化",
                        "使用1层BiLSTM编码,使用1dCNN解码",
                        "使用没有经过stride的14×14的patch的conv-n",
                        "使用image和sequence的序列到图像patch注意力对齐,返回cat",
                        "注意力中间添加0.2的dropout",
                        "不！使用image和sequence的预训练参数preload",
                        "初始学习率为1e-3",
                        "不使用重投影。FCProject: 无非线性激活,dim=64"
                    ],
                    'model': {
                        'fusion': {
                            'type': 'seq_attend_img_cat'
                        },
                        'conv_backbone': {
                            'params': {
                                'conv-n': {
                                    'strides': [1,1,1,1]
                                }
                            }
                        }
                    }
                })
machine.addTask('train',
                {
                    'task': {
                        'version': 64,
                        'preload_state_dict_versions': [15,16]
                    },
                    'description': [
                        "使用300维度的GloVe初始化",
                        "序列长度为300",
                        "提前终止的标准为loss",
                        "使用sgd优化",
                        "使用1层BiLSTM编码,使用1dCNN解码",
                        "使用没有经过stride的14×14的patch的conv-n",
                        "使用image和sequence的序列到图像patch注意力对齐,返回cat",
                        "注意力中间添加0.2的dropout",
                        "使用image和sequence的预训练参数preload",
                        "初始学习率为1e-3",
                        "不使用重投影。FCProject: 无非线性激活,dim=64"
                    ],
                    'model': {
                        'fusion': {
                            'type': 'seq_attend_img_cat'
                        },
                        'conv_backbone': {
                            'params': {
                                'conv-n': {
                                    'strides': [1,1,1,1]
                                }
                            }
                        }
                    }
                })
machine.addTask('train',
                {
                    'task': {
                        'version': 65,
                        'preload_state_dict_versions': []
                    },
                    'description': [
                        "使用300维度的GloVe初始化",
                        "序列长度为300",
                        "提前终止的标准为loss",
                        "使用sgd优化",
                        "使用1层BiLSTM编码,使用1dCNN解码",
                        "使用经过了stride的7×7的patch的conv-n",
                        "使用image和sequence的序列到图像patch注意力对齐,返回cat",
                        "注意力中间添加0.2的dropout",
                        "不！使用image和sequence的预训练参数preload",
                        "初始学习率为1e-3",
                        "不使用重投影。FCProject: 无非线性激活,dim=64"
                    ],
                    'model': {
                        'fusion': {
                            'type': 'seq_attend_img_cat'
                        },
                        'conv_backbone': {
                            'params': {
                                'conv-n': {
                                    'strides': [2,1,1,1]
                                }
                            }
                        }
                    }
                })
machine.addTask('train',
                {
                    'task': {
                        'version': 66,
                        'preload_state_dict_versions': [14,15]
                    },
                    'description': [
                        "使用300维度的GloVe初始化",
                        "序列长度为300",
                        "提前终止的标准为loss",
                        "使用sgd优化",
                        "使用1层BiLSTM编码,使用1dCNN解码",
                        "使用经过了stride的7×7的patch的conv-n",
                        "使用image和sequence的序列到图像patch注意力对齐,返回cat",
                        "注意力中间添加0.2的dropout",
                        "使用image和sequence的预训练参数preload",
                        "初始学习率为1e-3",
                        "不使用重投影。FCProject: 无非线性激活,dim=64"
                    ],
                    'model': {
                        'fusion': {
                            'type': 'seq_attend_img_cat'
                        },
                        'conv_backbone': {
                            'params': {
                                'conv-n': {
                                    'strides': [2,1,1,1]
                                }
                            }
                        }
                    }
                })

machine.addTask('test',
                {
                    "task": {
                        "version": 59
                    }
                })
machine.addTask('test',
                {
                    "task": {
                        "version": 60
                    }
                })
machine.addTask('test',
                {
                    "task": {
                        "version": 61
                    }
                })
machine.addTask('test',
                {
                    "task": {
                        "version": 62
                    }
                })
machine.addTask('test',
                {
                    "task": {
                        "version": 63
                    }
                })
machine.addTask('test',
                {
                    "task": {
                        "version": 64
                    }
                })
machine.addTask('test',
                {
                    "task": {
                        "version": 65
                    }
                })
machine.addTask('test',
                {
                    "task": {
                        "version": 66
                    }
                })

machine.execute()
