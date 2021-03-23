from autorun.machine import ExecuteMachine

machine = ExecuteMachine()
machine.addTask('train',
                {
                    "task": {
                        "version": 47
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
                        "FCProject: 非线性激活,dim=64"
                    ],
                    "model": {
                        "fusion": {
                            "params": {
                                "bili_norm_type": "ln",
                                "bili_affine": False,
                                "bili_non_linear": None
                            }
                        },
                        "reproject": {
                            "params": {
                                "out_dim": 64
                            }
                        }
                    }
                })
machine.addTask('train',
                {
                    "task": {
                        "version": 48
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
                        "FCProject: 非线性激活,dim=64"
                    ],
                    "model": {
                        "fusion": {
                            "params": {
                                "bili_norm_type": "ln",
                                "bili_affine": True,
                                "bili_non_linear": None
                            }
                        },
                        "reproject": {
                            "params": {
                                "out_dim": 64
                            }
                        }
                    }
                })
machine.addTask('train',
                {
                    "task": {
                        "version": 49
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
                        "FCProject: 非线性激活,dim=64"
                    ],
                    "model": {
                        "fusion": {
                            "params": {
                                "bili_norm_type": "ln",
                                "bili_affine": False,
                                "bili_non_linear": "tanh"
                            }
                        },
                        "reproject": {
                            "params": {
                                "out_dim": 64
                            }
                        }
                    }
                })
machine.addTask('train',
                {
                    "task": {
                        "version": 50
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
                        "FCProject: 非线性激活,dim=64"
                    ],
                    "model": {
                        "fusion": {
                            "params": {
                                "bili_norm_type": "ln",
                                "bili_affine": True,
                                "bili_non_linear": "tanh"
                            }
                        },
                        "reproject": {
                            "params": {
                                "out_dim": 64
                            }
                        }
                    }
                })
machine.addTask('test',
                {
                    "task": {
                        "version": 47
                    }
                })
machine.addTask('test',
                {
                    "task": {
                        "version": 48
                    }
                })
machine.addTask('test',
                {
                    "task": {
                        "version": 49
                    }
                })
machine.addTask('test',
                {
                    "task": {
                        "version": 50
                    }
                })
machine.execute()
