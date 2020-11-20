import os
import shutil
from tqdm import tqdm

from utils.file import loadJson
from utils.magic import sample

###############################################
# 从未分类的数据文件中按照家族log文件进行每个家族的
# 数据文件采样。采样后分类到文件夹中
###############################################
def sampleClassWiseData(dst_path,
                        log_file_path,
                        num_per_class=20):

    family_report = loadJson(log_file_path)

    for fname, flist in tqdm(family_report.items()):
        if len(flist) >= num_per_class:
            os.mkdir(dst_path+fname)
            cans = sample(flist, num_per_class)

            for can in cans:
                can_fname = can.split('/')[-1]
                shutil.copy(can, dst_path+fname+'/'+can_fname)


if __name__ == '__main__':
    sampleClassWiseData(dst_path='E:/LargePE-API-raw/families/Per-50/',
                        log_file_path='E:/LargePE-API-raw/reports/class_stat_log.json',
                        num_per_class=50)