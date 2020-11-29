import os

##########################################################
# 本函数用于创建数据集的文件夹结构
##########################################################
def makeDatasetDirStruct(base_path):
    assert base_path[-1] == '/'

    if not os.path.exists(base_path):
        os.mkdir(base_path)

    os.mkdir(base_path + 'all/')
    os.mkdir(base_path + 'all/api/')
    os.mkdir(base_path + 'all/img/')
    os.mkdir(base_path + 'train/')
    os.mkdir(base_path + 'train/api/')
    os.mkdir(base_path + 'train/img/')
    os.mkdir(base_path + 'validate/')
    os.mkdir(base_path + 'validate/api/')
    os.mkdir(base_path + 'validate/img/')
    os.mkdir(base_path + 'test/')
    os.mkdir(base_path + 'test/api/')
    os.mkdir(base_path + 'test/img/')

    os.mkdir(base_path + 'PEs/')
    os.mkdir(base_path + 'models/')

    os.mkdir(base_path + 'data/')
    os.mkdir(base_path + 'data/train/')
    os.mkdir(base_path + 'data/validate/')
    os.mkdir(base_path + 'data/test/')
    os.mkdir(base_path + 'doc/')

    print('Done')
