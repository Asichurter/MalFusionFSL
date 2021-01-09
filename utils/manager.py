import warnings
import config
import numpy as np
import torch as t

from utils.file import dumpJson
from utils.os import joinPath
from utils.timer import StepTimer

##########################################################
# 项目数据集路径管理器。
# 可以使用方法获取数据集路径。
# d_type用于指定当前使用的是训练集，验证集还是测试集，指定为all时代表
# 三者均不是。
##########################################################
class PathManager:

    def __init__(self, dataset, subset='all', version=None, model_name=None):
        self.Base = config.env.DatasetBasePath
        self.Dataset = dataset
        self.Subset = subset
        self.Version = version
        self.ModelName = model_name

    # 所有数据集的根目录
    def base(self):
        return joinPath(self.Base, is_dir=True)

    # 数据集的根目录
    def datasetBase(self):
        return joinPath(self.Base, self.Dataset, is_dir=True)

    # 存储子集api数据的文件夹目录
    def apiDataFolder(self):
        return joinPath(self.Base, self.Dataset, self.Subset, 'api', is_dir=True)

    # 存储子集img数据的文件夹目录
    def imgDataFolder(self):
        return joinPath(self.Base, self.Dataset, self.Subset, 'img', is_dir=True)

    # 预训练的词嵌入的存储路径
    def wordEmbedding(self):
        return joinPath(self.Base, self.Dataset, 'data/matrix.npy')

    # 预训练的词嵌入的下标映射
    def wordIndex(self):
        return joinPath(self.Base, self.Dataset, 'data/word_map.json')

    # 子集的api数据矩阵存储路径
    def apiData(self):
        if self.Subset == 'all':
            raise ValueError("[PathManager] 类型为all时无api数据文件")
        return joinPath(self.Base, self.Dataset, 'data', self.Subset, 'api.npy')

    # 子集的img数据矩阵存储路径
    def imgData(self):
        if self.Subset == 'all':
            raise ValueError("[PathManager] 类型为all时无pe数据文件")
        return joinPath(self.Base, self.Dataset, 'data', self.Subset, 'img.npy')

    # 子集的api数据序列长度json存储路径
    def apiSeqLen(self):
        if self.Subset == 'all':
            raise ValueError("[PathManager] 类型为all时无api序列长度数据文件")
        return joinPath(self.Base, self.Dataset, 'data', self.Subset, 'seq_length.json')

    # 子集的类下标到类名称的映射存储路径
    def subsdetIdxClassMapper(self):
        if self.Subset == 'all':
            raise ValueError("[PathManager] 类型为all时无“下标-类”映射数据文件")
        return joinPath(self.Base, self.Dataset, 'data', self.Subset, 'idx_mapping.json')

    # 返回model存储路径
    def model(self):
        return joinPath(self.Base, self.Dataset, 'models', self.ModelName)

    # 返回当前version的doc存储目录
    def doc(self):
        return joinPath(self.Base, self.Dataset, 'doc', str(self.Version), is_dir=True)

    # 存储训练时数据的文件路径
    def trainStat(self):
        return joinPath(self.Base, self.Dataset, 'doc', str(self.Version), "train_stat.json")

    # 存储测试时统计数据的文件路径
    def testStat(self):
        return joinPath(self.Base, self.Dataset, 'doc', str(self.Version), "test_stat.json")

    # doc的根目录
    def docBase(self):
        return joinPath(self.Base, self.Dataset, 'doc', is_dir=True)

    # data的根目录
    def dataBase(self):
        return joinPath(self.Base, self.Dataset, 'data', is_dir=True)

    # 子集的data根目录
    def subDataBase(self):
        return joinPath(self.Base, self.Dataset, 'data', self.Subset, is_dir=True)


#########################################
# 训练时的统计数据管理器。主要用于记录训练和验证的
# 正确率和损失值，并且根据选择的标准来选择最优模型
# 保存和打印训练数据。在记录验证数据时会打印训练和
# 验证信息
#
# 实质上不直接用于统计，而是作为统计器内核，被训
# 练/测试的统计器进行调用
#########################################
class StatKernel:
    def __init__(self,
                 report_iter=100,
                 metric_num=1,
                 criteria = "metric",
                 criteria_metric_index=0,
                 metric_names=['acc']):

        self.MetricHist = []
        self.LossHist = []
        self.MetricNum = metric_num
        self.MetricNames = metric_names
        self.CriteriaMetricIndex = criteria_metric_index
        self.Criteria = criteria
        self.BestVal = float('inf') if criteria=='loss' else -1.
        self.BestValEpoch = -1
        self.ReportIter = report_iter

        self.RecMetricCache = None
        self.RecLossCache = None

        if criteria not in ['metric', 'loss']:
            warnings.warn("没有指定保存模型的Criteria，默认为总是保存最新，即关闭提前停止")

    def record(self, metric, loss):
        # 适配np的ndarray
        if isinstance(metric, np.ndarray):
            metric = metric.tolist()
        self.MetricHist += metric           # 对于多个metric，进行展平处理，存放在同一个list中
        self.LossHist.append(loss)

    def update(self, current_epoch):
        '''
        检查指标是否超过了best指标，是否可以更新模型
        :param current_epoch: 进行检查时的epoch
        :return: 指标是否被更新
        '''
        if self.Criteria == 'metric':
            best_recent_val = self.getRecentMetric()[self.CriteriaMetricIndex]          # 使用index定位一个用于保存的唯一metric
            if best_recent_val > self.BestVal:
                self.BestVal = best_recent_val
                self.BestValEpoch = current_epoch
                return True
            else:
                return False

        elif self.Criteria == 'loss':
            best_recent_loss = self.getRecentLoss()
            if best_recent_loss < self.BestVal:
                self.BestVal = best_recent_loss
                self.BestValEpoch = current_epoch
                return True
            else:
                return False

        # 默认为保存最新模型
        return True

    def printRecent(self, title, all_time=False, cache_recent=True):
        recent_metric = self.getRecentMetric()
        recent_loss = self.getRecentLoss()
        for i,name in enumerate(self.MetricNames):
            print(f"{title} {name}: {recent_metric[i]}")
        print(title, 'loss:', recent_loss)

        if all_time:
            self.printAllTime(title='Current '+title)

        if cache_recent:
            self.RecMetricCache = recent_metric
            self.RecLossCache = recent_loss

    def printAllTime(self, title):
        print(title, end=' :')
        all_time_metric = self.getAlltimeMetric()
        all_time_loss = self.getAlltimeLoss()
        for i, name in enumerate(self.MetricNames):
            print(f"{name}:{all_time_metric[i]}", end=" ")
        print()
        print(title, 'loss:', all_time_loss)

    def printBest(self, title):
        print(f"Best {title} {self.MetricNames[self.CriteriaMetricIndex]}: {self.BestVal} (at {self.BestValEpoch} epoch)")

    def getRecentMetric(self):
        '''
        计算最近的metric值（根据report_iter计算）
        '''
        n = self.MetricNum
        iter_len = self.ReportIter
        return [np.mean(self.MetricHist[-n*iter_len+i::n]) for i in range(n)]       # 取出最后iter_len个指标的平均值

    def getRecentLoss(self):
        '''
        计算最近的loss值（根据report_iter计算）
        :return:
        '''
        iter_len = self.ReportIter
        return np.mean(self.LossHist[-iter_len:])

    def getAlltimeMetric(self):
        n = self.MetricNum
        return [np.mean(self.MetricHist[i::n]) for i in range(n)]

    def getAlltimeLoss(self):
        return np.mean(self.LossHist)


class TrainStatManager:
    def __init__(self,
                 stat_save_path=None,
                 model_save_path=None,
                 save_latest_model=False,
                 train_report_iter=100,
                 val_report_iter=50,
                 total_iter=50000,
                 metric_num=1,
                 criteria = "metric",
                 criteria_metric_index=0,
                 metric_names=['Acc']):

        self.TrainStat = StatKernel(train_report_iter, metric_num, metric_names=metric_names)
        self.ValStat = StatKernel(val_report_iter, metric_num, criteria, criteria_metric_index, metric_names)

        self.StatSavePath = stat_save_path
        self.ModelSavePath = model_save_path
        self.SaveLastestModelFlag = save_latest_model
        self.Timer = StepTimer(total_steps=total_iter)

        self.TrainIterCount = 0
        self.ValIterCount = 0
        self.TrainReportIter = train_report_iter
        self.ValReportIter = val_report_iter

    def begin(self):
        self._printNextTip()
        self.Timer.begin()

    def recordTrain(self, metric, loss):
        self.TrainStat.record(metric, loss)
        self.TrainIterCount += 1

    # 记录每一次validate的结果，自动化判断是否val是否结束进行打印
    # 如果记录的总次数到达了report节点，便会调用更新检查，并打印出最近数据
    def recordVal(self, metric, loss, model):
        self.ValStat.record(metric, loss)
        self.ValIterCount += 1

        if self.ValIterCount % self.ValReportIter == 0:
            updated = self.ValStat.update(self.TrainIterCount)

            if updated and self.ModelSavePath is not None:
                t.save(model.state_dict(), self.ModelSavePath)
            if self.SaveLastestModelFlag and self.ModelSavePath is not None:
                t.save(model.state_dict(), self.ModelSavePath+'_latest')

            self._printBlockSeg()
            self.TrainStat.printRecent(title='Train', all_time=False)
            self._printSectionSeg()
            self.ValStat.printRecent(title='Val', all_time=False)
            self.ValStat.printBest(title='Val')
            self._printSectionSeg()
            self.Timer.step(step_stride=self.TrainReportIter, prt=True, end=False)
            self._printBlockSeg()
            self._printNextTip()

    def dumpStatHist(self):
        res = {
            'train': {
                'metrics': self.TrainStat.MetricHist,
                'loss': self.TrainStat.LossHist
            },
            'validate': {
                'metrics': self.ValStat.MetricHist,
                'loss': self.ValStat.LossHist
            }
        }
        if self.StatSavePath is not None:
            dumpJson(res, self.StatSavePath)

    def _printSectionSeg(self):
        print('----------------------------------')

    def _printBlockSeg(self):
        print('***********************************')

    def _printNextTip(self):
        print('%d -> %d epoches...\n\n' % (self.TrainIterCount, self.TrainIterCount + self.TrainReportIter))
        # 打印运行任务的摘要
        config.printRunConfigSummary()

    # 用于绘制总体图时使用的压缩hist方法
    def getHistMetric(self, idx=0):
        th = self.TrainStat.MetricHist[idx::self.TrainStat.MetricNum]
        vh = self.ValStat.MetricHist[idx::self.ValStat.MetricNum]
        th,vh = np.array(th), np.array(vh)
        th = np.mean(th.reshape(-1, self.TrainReportIter), axis=1)
        vh = np.mean(th.reshape(-1, self.ValReportIter), axis=1)
        return th,vh

    # 用于绘制总体图时使用的压缩hist方法
    def getHistLoss(self):
        th = self.TrainStat.LossHist
        vh = self.ValStat.LossHist
        th, vh = np.array(th), np.array(vh)
        th = np.mean(th.reshape(-1, self.TrainReportIter), axis=1)
        vh = np.mean(th.reshape(-1, self.ValReportIter), axis=1)
        return th, vh

    # 该方法使用cache，因此必须保证在printRecent之后调用
    # 主要用于实时的metric的plot
    def getRecentRecord(self, metric_idx=0):
        return self.TrainStat.RecMetricCache[metric_idx], self.TrainStat.RecLossCache, \
               self.ValStat.RecMetricCache[metric_idx], self.ValStat.RecLossCache


class TestStatManager:
    def __init__(self,
                 stat_save_path=None,
                 test_report_iter=100,
                 total_iter=50000,
                 metric_num=1,
                 metric_names=['Acc']):

        self.TestStat = StatKernel(test_report_iter, metric_num, metric_names=metric_names)
        self.StatSavePath = stat_save_path
        self.Timer = StepTimer(total_steps=total_iter)
        self.TestIterCount = 0
        self.TotalIter = total_iter
        self.ReportIter = test_report_iter

    def begin(self):
        self.Timer.begin()

    def recordTest(self, metric, loss):
        self.TestStat.record(metric, loss)
        self.TestIterCount += 1

        if self.TestIterCount == self.TotalIter:
            self._printBlockSeg()
            print('Final Statistics:')
            self.TestStat.printAllTime(title="Final Test")
            self.Timer.step(step_stride=self.ReportIter, prt=True, end=True)

        elif self.TestIterCount % self.ReportIter == 0:
            print(self.TestIterCount, "Epoch")
            self.TestStat.printRecent(title="Test", all_time=True)
            self.Timer.step(step_stride=self.ReportIter, prt=True)

    def _printBlockSeg(self):
        print('\n\n****************************************')