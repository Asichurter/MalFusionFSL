import config
from utils.os import joinPath

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
            raise ValueError("类型为all时无api数据文件")
        return joinPath(self.Base, self.Dataset, 'data', self.Subset, 'api.npy')

    # 子集的img数据矩阵存储路径
    def imgData(self):
        if self.Subset == 'all':
            raise ValueError("类型为all时无pe数据文件")
        return joinPath(self.Base, self.Dataset, 'data', self.Subset, 'img.npy')

    # 子集的api数据序列长度json存储路径
    def apiSeqLen(self):
        if self.Subset == 'all':
            raise ValueError("类型为all时无api序列长度数据文件")
        return joinPath(self.Base, self.Dataset, 'data', self.Subset, 'seq_length.json')

    # 子集的类下标到类名称的映射存储路径
    def subsdetIdxClassMapper(self):
        if self.Subset == 'all':
            raise ValueError("类型为all时无“下标-类”映射数据文件")
        return joinPath(self.Base, self.Dataset, 'data', self.Subset, 'idx_mapping.json')

    # 返回model存储路径
    def model(self):
        return joinPath(self.Base, 'models', self.ModelName)

    # 返回当前version的doc存储目录
    def doc(self):
        return joinPath(self.Base, 'doc', self.Version, is_dir=True)

    # doc的根目录
    def docBase(self):
        return joinPath(self.Base, 'doc', is_dir=True)

    # data的根目录
    def dataBase(self):
        return joinPath(self.Base, 'data', is_dir=True)

    # 子集的data根目录
    def subDataBase(self):
        return joinPath(self.Base, 'data', self.Subset, is_dir=True)