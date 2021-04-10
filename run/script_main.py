from utils.summary import makeResultSummaryByVerRange, makeResultByTrainConfigCond

# makeResultSummaryByVerRange(dataset='virushare-20',
#                             version_range=[80, 100])
makeResultByTrainConfigCond(dataset='virushare-20',
                            train_config_cond={
                                'model': {
                                    'model_name': 'SIMPLE',
                                    'fusion': {
                                        'type': 'cat'
                                    }
                                }
                            })
