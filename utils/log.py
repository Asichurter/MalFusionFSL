from utils.file import dumpJson

#######################################################
# 用于记录运行时的警告和错误信息，并打印或者转储文件
# 发生错误或者警告时，需要输入一个引发错误或者警告的实体entity
# 每一轮中如果没有发生错误，调用log_success
#######################################################
class Reporter:
    def __init__(self):
        self.ErrorList = []
        self.WarningList = []
        self.SuccessCount = 0
        self.TotalCount = 0

    def logSuccess(self):
        self.SuccessCount += 1
        self.TotalCount += 1

    def logError(self, entity, msg):
        self.ErrorList.append([entity, msg])
        self.TotalCount += 1

    def logWarning(self, entity, msg):
        self.WarningList.append([entity, msg])
        self.TotalCount += 1

    def report(self):
        print('\n\n----------------Report--------------------')
        print('Total: %d   Success: %d   Error: %d   Warning: %d'  %
              (self.TotalCount, self.SuccessCount, len(self.ErrorList), len(self.WarningList)))

        if len(self.WarningList) > 0:
            print('\nWarnings:')
            for idx,i in enumerate(self.WarningList):
                print('#%d'%idx, '%s: %s'%(i[0], i[1]))

        if len(self.ErrorList) > 0:
            print('\nErrors:')
            for idx,i in enumerate(self.ErrorList):
                print('#%d'%idx, '%s: %s'%(i[0], i[1]))

    def dump(self, path):
        dump_file = {'errors': {}, 'warnings': {}}

        # 将entity作为键，信息内容作为值
        for e in self.ErrorList:
            dump_file['errors'][e[0]] = e[1]

        for w in self.WarningList:
            dump_file['warnings'][w[0]] = w[1]

        dumpJson(dump_file, path)