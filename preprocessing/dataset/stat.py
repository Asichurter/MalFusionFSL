import os
from tqdm import tqdm

from utils.file import loadJson, dumpIterable, dumpJson
from utils.general import datasetTraverse

#####################################################
# 统计数据集中存在API调用序列且序列长度满足要求的文件
#####################################################
def statValidJsonReport(dir_path, len_thresh=10,
                        class_dir=False,
                        name_prefix=None,
                        dump_valid_path=None):

    valid = invalid = too_short = total = 0
    valid_list = []

    for folder in os.listdir(dir_path):
        folder_path = dir_path+folder+'/'
        if class_dir:
            items = os.listdir(folder_path)
        else:
            items = [name_prefix+'.json']

        for item in items:
            total_length = 0
            total += 1
            print('#%d'%total, folder_path+item, end=': ')

            try:
                report = loadJson(folder_path+item)
                raw_file_name = report['target']['file']['name']
                for process in report['behavior']['processes']:
                    total_length += len(process['calls'])

                if total_length < len_thresh:
                    too_short += 1
                    print('too short:', total_length)
                else:
                    valid += 1
                    valid_list.append({'file':raw_file_name,
                                       'len':total_length,
                                       'rawPath':folder_path+item})
                    print('valid')
            except Exception as e:
                invalid += 1
                print('Error: ', str(e))

    print('Total:', total)
    print('Valid:', valid)
    print('Invalid:', invalid)
    print('Too Short:', too_short)

    if dump_valid_path is not None:
        dumpIterable(valid_list, title='valid_file_name', path=dump_valid_path)

#####################################################
# 统计数据集中存在__exception__的数据集文件数量
#####################################################
def statExceptionReport(dir_path, class_dir=False,
                        name_prefix=None,
                        dump_noexp_path=None):

    def statExceptionReportInner(count_, filep_, report_, list_, dict_):
        print('# %d'%count_, filep_, end=' ')

        if len(dict_) == 0:
            dict_ = {
                'noexc': 0,
                'exc': 0,
                'err': 0
            }

        for api in report_['apis']:
            if api == '__exception__':
                dict_['exc'] += 1
                list_.append({'file': filep_,
                              'hasExc': True})
                print('Exception')
                return list_, dict_
        dict_['noexc'] += 1
        list_.append({'file': filep_,
                      'hasExc': False})
        print('Normal')
        return list_,dict_

    def statExceptionReportFcb(e, list_, dict_):
        dict_['err'] += 1
        print("Error")

    def statExceptionReportFNcb(reporter_, list_, dict_):
        print('*'*50)
        print("Total:", dict_['noexc']+dict_['exc']+dict_['err'])
        print("No Exception:", dict_['noexc'])
        print('Exception:', dict_['exc'])
        print('Error:', dict_['err'])
        print('*' * 50)

        if dump_noexp_path is not None:
            dumpIterable(list_, title='exception_check_results',
                         path=dump_noexp_path)


    datasetTraverse(dir_path=dir_path,
                    exec_kernel=statExceptionReportInner,
                    class_dir=class_dir,
                    name_prefix=name_prefix,
                    success_callback=lambda x,y:None,         # 不做success的默认打印
                    fail_callback=statExceptionReportFcb,
                    final_callback=statExceptionReportFNcb)

#####################################################
# 统计数据集中按照name划分的family的规模情况
#####################################################
def statMalClassesOnNames(dir_path,
                          # log_file_path=None,
                          # log_list_key=None,               # list型log的list键值
                          # log_item_path_key=None,          # list型log的每一个item需要的元素的键值
                          dump_log_path=None):


    def statMalClassesOnNamesInner(count_, filep_, report_, list_, dict_):
        print('# %d'%count_, filep_, end=' ')

        family = '.'.join(report_['name'].split('.')[:3])       # family是name中点隔开的前三个字段
        if family not in dict_:
            dict_[family] = [filep_]
        else:
            dict_[family].append(filep_)            # family键对应的是对应的数据文件的path
        return list_, dict_

    def statMalClassesOnNamesFNcb(reporter_, list_, dict_):
        for f,c in dict_.items():
            print(f,len(c))

        if dump_log_path is not None:
            dumpJson(dict_, dump_log_path)

    datasetTraverse(dir_path=dir_path,
                    exec_kernel=statMalClassesOnNamesInner,
                    class_dir=True,                     # 所有report必须存在于一个上级文件夹单独存在的目录中，即唯一class
                    final_callback=statMalClassesOnNamesFNcb)

if __name__ == '__main__':
    # statValidJsonReport(dir_path='F:/result2/cuckoo/analyses/',
    #                     class_dir=False,
    #                     name_prefix='reports/report',
    #                     dump_valid_path='D:/datasets/LargePE-API-raw/reports/valid_file_list.json')
    # statExceptionReport(dir_path='E:/LargePE-API-raw/extracted/',
    #                     class_dir=True,
    #                     dump_noexp_path='E:/LargePE-API-raw/reports/exception_stat_log.json')
    statMalClassesOnNames(dir_path='E:/LargePE-API-raw/extracted/',
                          dump_log_path='E:/LargePE-API-raw/reports/class_stat_log.json')