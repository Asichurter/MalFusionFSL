from autorun.machine import ExecuteMachine

machine = ExecuteMachine()
machine.addTask('train',
                {
                    "task": {
                        "version": 39
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
                        "使用image和sequence的hdm双线性层,不使用仿射函数，不使用激活函数，bn标准化",
                        "初始学习率为1e-3",
                        "FCProject: 非线性激活,dim=64"
                    ],
                    "model": {
                        "fusion": {
                            "params": {
                                "bili_norm_type": "bn",
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
                        "version": 40
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
                        "FCProject: 非线性激活,dim=64"
                    ],
                    "model": {
                        "fusion": {
                            "params": {
                                "bili_norm_type": "bn",
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
                        "version": 41
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
                        "使用image和sequence的hdm双线性层,不使用仿射函数，使用tanh激活函数，bn标准化",
                        "初始学习率为1e-3",
                        "FCProject: 非线性激活,dim=64"
                    ],
                    "model": {
                        "fusion": {
                            "params": {
                                "bili_norm_type": "bn",
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
                        "version": 42
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
                        "使用image和sequence的hdm双线性层,使用仿射函数，使用tanh激活函数，bn标准化",
                        "初始学习率为1e-3",
                        "FCProject: 非线性激活,dim=64"
                    ],
                    "model": {
                        "fusion": {
                            "params": {
                                "bili_norm_type": "bn",
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
                        "version": 39
                    }
                })
machine.addTask('test',
                {
                    "task": {
                        "version": 40
                    }
                })
machine.addTask('test',
                {
                    "task": {
                        "version": 41
                    }
                })
machine.addTask('test',
                {
                    "task": {
                        "version": 42
                    }
                })
machine.execute()
