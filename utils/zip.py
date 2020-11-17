import zipfile
import os
from tqdm import tqdm

from utils.log import Reporter

###################################################
# 将文件夹打包为zip文件
# src: 打包的文件夹
# target: 打包后的压缩文件
# log_dump_path: 打包过程中遇到的错误的报告文件存储地址
###################################################
def makeZipPack(src_path, target_path, log_dump_path=None):
    zip_file = zipfile.ZipFile(target_path, 'w', zipfile.ZIP_DEFLATED)
    reporter = Reporter()
    count = 0

    for path, dirNames, fileNames in tqdm(os.walk(src_path)):
        fpath = path.replace(src_path, '')
        for name in fileNames:
            fullName = os.path.join(path, name)#.decode(encoding='gbk')
            count += 1
            print('#', count)

            name = fpath + '/' + name
            try:
                zip_file.write(fullName, name)
                reporter.logSuccess()
            except Exception as e:
                reporter.logError(entity=fullName, msg=str(e))

    zip_file.close()
    reporter.report()
    if log_dump_path is not None:
        reporter.dump(log_dump_path)

if __name__ == '__main__':
    makeZipPack(src_path='F:/LargePE-API-raw/extracted/',
                target_path='F:/LargePE-API-raw/extracted.zip',
                log_dump_path='F:/LargePE-API-raw/reports/zip_json_log.json')