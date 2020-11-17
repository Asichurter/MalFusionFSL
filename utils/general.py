from utils.file import *
from utils.log import Reporter

############################################
# 遍历数据集文件结构的遍历包装器
#
# 该函数会遍历数据集中的JSON数据文件，同时读取文件
# 将其递交给kernel，同时在执行kernel同时维护一个
# 数据列表和数据字典。包装器还支持指定成功/失败处理
# 一个数据文件之后调用的回调函数以支持自定义打印，
# 同时还支持在遍历结束以后执行一个最终的回调函数进行
# 一些保存结果或打印统计数据的操作等
############################################
def datasetTraverse(dir_path,               # 数据集的根目录
                    exec_kernel,            # 遍历每个数据文件时要执行的操作
                                            # // 参数：(序号，文件路径，报告json，数据列表，数据字典)，必须返回数据列表和数据字典
                    class_dir=False,        # 是否是按类划分的数据集
                    name_prefix=None,       # 数据集文件的前缀，若为None则为文件夹名
                    name_suffix=None,       # 数据集文件的后缀，若为None则为.json
                    success_callback=None,  # 成功执行一次数据文件处理后的回调函数，用于打印信息等
                                            # // 参数：(数据列表，数据字典)
                    fail_callback=None,     # 执行一次数据文件处理失败后的回调函数，用于打印调试信息
                                            # // 参数：(异常对象, 数据列表，数据字典)
                    final_callback=None):   # 遍历完成之后的回调函数，用于保存信息
                                            # // 参数：(reporter对象，数据列表，数据字典)

    count = 0
    reporter = Reporter()
    stat_list = []          # 执行过程中维护的数据列表
    stat_dict = {}          # 执行过程中维护的数据字典

    for folder in os.listdir(dir_path):
        folder_path = dir_path+folder+'/'
        if class_dir:
            items = os.listdir(folder_path)
        else:
            prefix = name_prefix if name_prefix is not None else folder
            suffix = name_suffix if name_suffix is not None else '.json'
            items = [prefix+suffix]

        for item in items:
            count += 1

            try:
                report = loadJson(folder_path+item)
                stat_list, stat_dict = exec_kernel(count,
                                                   folder_path+item,
                                                   report,
                                                   stat_list,
                                                   stat_dict)

                reporter.logSuccess()
                if success_callback is not None:
                    success_callback(stat_list, stat_dict)
            except Exception as e:
                reporter.logError(entity=folder_path+item,
                                  msg=str(e))
                if fail_callback is not None:
                    fail_callback(e, stat_list, stat_dict)

    reporter.report()
    if final_callback is not None:
        final_callback(reporter, stat_list, stat_dict)
