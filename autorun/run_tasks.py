from autorun.machine import ExecuteMachine

machine = ExecuteMachine()
machine.addTask('train',
                {
                    "task": {
                        "version": 34
                    },
                    "training": {
                        "epoch": 30000
                    },
                    "description": [
                        "使用300维度的GloVe初始化",
                        "序列长度为300",
                        "提前终止的标准为loss",
                        "使用sgd优化",
                        "使用1层BiLSTM编码,使用1dCNN解码",
                        "使用image和sequence的hdm双线性层,使用仿射函数，不使用激活函数，bn标准化",
                        "初始学习率为1e-3",
                        "FCProject: 非线性激活,dim=128"
                    ],
                    "model": {
                        "fusion": {
                            "params": {
                                "bili_norm_type": "bn",
                                "bili_affine": True,
                                "bili_non_linear": None
                            }
                        }
                    }
                })
machine.addTask('train',
                {
                    "task": {
                        "version": 35
                    },
                    "training": {
                        "epoch": 30000
                    },
                    "description": [
                        "使用300维度的GloVe初始化",
                        "序列长度为300",
                        "提前终止的标准为loss",
                        "使用sgd优化",
                        "使用1层BiLSTM编码,使用1dCNN解码",
                        "使用image和sequence的hdm双线性层,不使用仿射函数，不使用激活函数，ln标准化",
                        "初始学习率为1e-3",
                        "FCProject: 非线性激活,dim=128"
                    ],
                    "model": {
                        "fusion": {
                            "params": {
                                "bili_norm_type": "ln",
                                "bili_affine": False,
                                "bili_non_linear": None
                            }
                        }
                    }
                })
machine.addTask('train',
                {
                    "task": {
                        "version": 36
                    },
                    "training": {
                        "epoch": 30000
                    },
                    "description": [
                        "使用300维度的GloVe初始化",
                        "序列长度为300",
                        "提前终止的标准为loss",
                        "使用sgd优化",
                        "使用1层BiLSTM编码,使用1dCNN解码",
                        "使用image和sequence的hdm双线性层,不使用仿射函数，使用tanh激活函数，ln标准化",
                        "初始学习率为1e-3",
                        "FCProject: 非线性激活,dim=128"
                    ],
                    "model": {
                        "fusion": {
                            "params": {
                                "bili_norm_type": "ln",
                                "bili_affine": False,
                                "bili_non_linear": "tanh"
                            }
                        }
                    }
                })
machine.addTask('train',
                {
                    "task": {
                        "version": 37
                    },
                    "training": {
                        "epoch": 30000
                    },
                    "description": [
                        "使用300维度的GloVe初始化",
                        "序列长度为300",
                        "提前终止的标准为loss",
                        "使用sgd优化",
                        "使用1层BiLSTM编码,使用1dCNN解码",
                        "使用image和sequence的hdm双线性层,使用仿射函数，不使用激活函数，ln标准化",
                        "初始学习率为1e-3",
                        "FCProject: 非线性激活,dim=128"
                    ],
                    "model": {
                        "fusion": {
                            "params": {
                                "bili_norm_type": "ln",
                                "bili_affine": True,
                                "bili_non_linear": None
                            }
                        }
                    }
                })
machine.addTask('train',
                {
                    "task": {
                        "version": 38
                    },
                    "training": {
                        "epoch": 30000
                    },
                    "description": [
                        "使用300维度的GloVe初始化",
                        "序列长度为300",
                        "提前终止的标准为loss",
                        "使用sgd优化",
                        "使用1层BiLSTM编码,使用1dCNN解码",
                        "使用image和sequence的hdm双线性层,使用仿射函数，使用tanh激活函数，ln标准化",
                        "初始学习率为1e-3",
                        "FCProject: 非线性激活,dim=128"
                    ],
                    "model": {
                        "fusion": {
                            "params": {
                                "bili_norm_type": "ln",
                                "bili_affine": True,
                                "bili_non_linear": "tanh"
                            }
                        }
                    }
                })
machine.addTask('test',
                {
                    "task": {
                        "version": 34
                    }
                })
machine.addTask('test',
                {
                    "task": {
                        "version": 35
                    }
                })
machine.addTask('test',
                {
                    "task": {
                        "version": 36
                    }
                })
machine.addTask('test',
                {
                    "task": {
                        "version": 37
                    }
                })
machine.addTask('test',
                {
                    "task": {
                        "version": 38
                    }
                })
machine.execute()
