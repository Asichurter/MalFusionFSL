import os

##########################################################
# 训练GloVe词嵌入矩阵
##########################################################
def trainGloVe(base_path, dataset, size=300, type='all'):
    print("Running GloVe in Python 2 env using Shell...")
    if os.system(f'python2 ../utils/GloVe.py -p {base_path} -d {dataset} -s {size} -t {type}') != 0:
        raise ValueError("[GloVe] Fail to run GloVe.py, check run path")
    print('GloVe Done')