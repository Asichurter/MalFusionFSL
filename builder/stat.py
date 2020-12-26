import config

from utils.manager import TrainStatManager, TestStatManager, PathManager

def buildStatManager(is_train,
                     path_manager: PathManager,
                     train_config: config.TrainingConfig=None,
                     test_config: config.TestConfig=None):

    if is_train:
        if train_config is None:
            train_config = config.train
        manager = TrainStatManager(stat_save_path=path_manager.trainStat(),
                                   model_save_path=path_manager.model(),
                                   save_latest_model=train_config.SaveLatest,
                                   train_report_iter=train_config.ValCycle,
                                   val_report_iter=train_config.ValEpisode,
                                   total_iter=train_config.TrainEpoch,
                                   metric_num=len(train_config.Metrics),
                                   criteria=train_config.Criteria,
                                   criteria_metric_index=0,
                                   metric_names=train_config.Metrics)
    else:
        if test_config is None:
            test_config = config.test
        manager = TestStatManager(stat_save_path=path_manager.testStat(),
                                  test_report_iter=test_config.ReportIter,
                                  total_iter=test_config.Epoch,
                                  metric_num=len(test_config.Metrics),
                                  metric_names=test_config.Metrics)

    return manager