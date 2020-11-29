import os
from tqdm import tqdm

from utils.file import loadJson, dumpIterable, dumpJson
from utils.general import datasetTraverse, jsonPathListLogTraverse

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
                        exception_call_patience=20,
                        dump_noexp_path=None):

    def statExceptionReportInner(count_, filep_, report_, list_, dict_, **kwargs):
        print('# %d'%count_, filep_, end=' ')

        if len(dict_) == 0:
            dict_ = {
                'noexc': 0,
                'exc': 0,
                'err': 0,
                'exc_list': [],
                'noexc_list': []
            }

        apis = report_['apis']
        for i in range(len(apis)):
            if apis[i] == '__exception__' and i+1 < len(apis):      # 只关注exception出现的位置
                if apis[i+1] == 'NtTerminateProcess' and i+2==len(apis):           # 如果exception发生以后立刻terminate且进程结束，检测成功
                    print('terminate', end=' ')
                elif apis[i+1] == '__exception__':              # 如果连续的exception出现
                    j = 1
                    flag = False

                    while i+j < len(apis):                      # 检测连续的exception是否超过了耐心值
                        if j == exception_call_patience:        # 连续的exception达到了耐心值，检测成功
                            flag = True
                            print('successive exceptions', end=' ')
                            break
                        elif apis[i+j] != '__exception__':
                            break
                        else:
                            j += 1

                    if not flag:
                        continue
                else:               # 其余所有情况都视为检测失败
                    continue

                dict_['exc'] += 1
                dict_['exc_list'].append(filep_)
                print('Exception')
                return list_, dict_
        dict_['noexc'] += 1
        dict_['noexc_list'].append(filep_)
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
            dumpJson({'has_exception': dict_['exc_list'],
                      'no_exception': dict_['noexc_list']},
                     dump_noexp_path)


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
def statMalClassesOnNames(exc_log_file_path=None,
                          exc_log_list_key=None,               # list型log的list键值
                          dump_log_path=None,
                          scale_stairs=[20,40,50,60,80,100]):


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

        counts = [0]*len(scale_stairs)
        for family, f_list in dict_.items():
            for i,s in enumerate(scale_stairs):
                if len(f_list) >= s:
                    counts[i] += 1

        for s,c in zip(scale_stairs, counts):
            print("More than %d items:"%s, c)


    jsonPathListLogTraverse(log_file_path=exc_log_file_path,
                            exec_kernel=statMalClassesOnNamesInner,
                            list_key=exc_log_list_key,
                            final_callback=statMalClassesOnNamesFNcb)


if __name__ == '__main__':
    # statValidJsonReport(dir_path='F:/result2/cuckoo/analyses/',
    #                     class_dir=False,
    #                     name_prefix='reports/report',
    #                     dump_valid_path='D:/datasets/LargePE-API-raw/reports/valid_file_list.json')
    # statExceptionReport(dir_path='E:/LargePE-API-raw/extracted/',
    #                     class_dir=True,
    #                     dump_noexp_path='E:/LargePE-API-raw/reports/exception_stat_log.json')
    statMalClassesOnNames(exc_log_file_path='E:/LargePE-API-raw/reports/exception_stat_log.json',
                          exc_log_list_key='no_exception',
                          dump_log_path='E:/LargePE-API-raw/reports/class_stat_log.json')