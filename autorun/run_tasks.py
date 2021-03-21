from autorun.machine import ExecuteMachine

machine = ExecuteMachine()
machine.addTask('train',
                {
                    "task": {
                        "version": 26
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
                        "使用image和sequence的双线性层,使用仿射函数，tanh激活，bn标准化",
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
                        }
                    }
                })
machine.addTask('train',
                {
                    "task": {
                        "version": 27
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
                        "使用image和sequence的双线性层,不使用仿射函数，tanh激活，ln标准化",
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
                        }
                    }
                })
machine.addTask('train',
                {
                    "task": {
                        "version": 28
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
                        "使用image和sequence的双线性层,使用仿射函数，不使用激活函数，ln标准化",
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
                        }
                    }
                })
machine.addTask('train',
                {
                    "task": {
                        "version": 29
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
                        "使用image和sequence的双线性层,使用仿射函数，tanh激活，ln标准化",
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
                        }
                    }
                })

machine.addTask('test',
                {
                    "task": {
                        "version": 26
                    },
                    "testing_epoch": 5000
                })
machine.addTask('test',
                {
                    "task": {
                        "version": 27
                    },
                    "testing_epoch": 5000
                })
machine.addTask('test',
                {
                    "task": {
                        "version": 28
                    },
                    "testing_epoch": 5000
                })
machine.addTask('test',
                {
                    "task": {
                        "version": 29
                    },
                    "testing_epoch": 5000
                })
machine.execute()
