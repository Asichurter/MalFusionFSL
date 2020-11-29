from utils.file import dumpJson
from utils.general import datasetTraverse
from utils.file import loadJson

def removeAPIRedundancy(dir_path,
                        class_dir=True):

    def removeAPIRedundancyInner(count_, filep_, report_, list_, dict_, **kwargs):
        print('# %d'%count_, end=' ')

        new_report = {key:val for key,val in report_.items()}
        new_apis = []
        base = 0
        apis = report_['apis']
        while base < len(apis):
            shift = 1
            while base+shift < len(apis) and apis[base+shift]==apis[base]:
                shift += 1
            new_apis.append(apis[base])
            base += shift
        new_report['apis'] = new_apis
        dumpJson(new_report, filep_)
        return list_, dict_

    datasetTraverse(dir_path=dir_path,
                    exec_kernel=removeAPIRedundancyInner,
                    class_dir=class_dir)
