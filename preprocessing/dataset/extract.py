import os

from utils.file import dumpJson, loadJson
from utils.general import datasetTraverse
from utils.log import Reporter


def extractAPISequenceFromRaw(dir_path, dst_path, log_dump_path=None):

    def extractAPISequenceFromRawInner(count_, file_path_, report_, list_, dict_):
        print("# %d"%count_, end=' ')
        new_report = {}
        new_report['sha1'] = report_['target']['file']['sha1']
        new_report['name'] = report_['target']['file']['name']
        new_report['sha256'] = report_['target']['file']['sha256']
        new_report['sha512'] = report_['target']['file']['sha512']
        md5 = new_report['md5'] = report_['target']['file']['md5']

        apis = []
        for process in report_['behavior']['processes']:
            for call in process['calls']:
                apis.append(call['api'])
        new_report['apis'] = apis

        dumpJson(new_report, dst_path+md5+'.json')
        return list_, dict_

    def extractAPISequenceFromRawFNcb(reporter_, list_, dict_):
        if log_dump_path is not None:
            reporter_.dump(log_dump_path)

    datasetTraverse(dir_path,
                    extractAPISequenceFromRawInner,
                    class_dir=False,
                    name_prefix='report',
                    name_suffix='.json',
                    final_callback=extractAPISequenceFromRawFNcb)


##########################################
# 根据执行stat之后的log，从cuckoo的报告文件夹
# 中提取有效的数据文件，同时转存其哈希值和api序列。
# 转存时会根据md5哈希值来重命名数据文件
##########################################
def extractAPISeqOnLog(pre_log_path, dst_path, log_dump_path=None):
    logs = loadJson(pre_log_path)
    reporter = Reporter()

    for i,item in enumerate(logs['valid_files']):
        print('#', i+1, end=' ')
        try:
            report_ = loadJson(item['rawPath'])

            new_report = {}
            new_report['sha1'] = report_['target']['file']['sha1']
            new_report['name'] = report_['target']['file']['name']
            new_report['sha256'] = report_['target']['file']['sha256']
            new_report['sha512'] = report_['target']['file']['sha512']
            md5 = new_report['md5'] = report_['target']['file']['md5']

            apis = []
            for process in report_['behavior']['processes']:
                for call in process['calls']:
                    apis.append(call['api'])
            new_report['apis'] = apis
            dumpJson(new_report, dst_path+md5+'.json')

            reporter.logSuccess()
            print("Success")
        except Exception as e:
            reporter.logError(entity=item['rawPath'],
                              msg=str(e))
            print("Error:", str(e))

    reporter.report()
    if log_dump_path is not None:
        reporter.dump(log_dump_path)


if __name__ == '__main__':
    extractAPISeqOnLog(pre_log_path='F:/LargePE-API-raw/reports/valid_file_list.json',
                       dst_path='F:/LargePE-API-raw/extracted/',
                       log_dump_path='F:/LargePE-API-raw/reports/api_extract_report.json')