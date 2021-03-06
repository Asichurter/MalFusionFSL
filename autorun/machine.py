import os
import time
from pprint import pprint
from copy import deepcopy

from utils.file import loadJson, dumpJson
from utils.manager import PathManager
from utils.os import joinPath
from utils.timer import TimeFormatter


##############################################
# 任务自动运行机。通过传递每一次运行时train/test config
# 的修改内容来达到控制任务参数的目的，同时使用命令行运行
# run目录下的train和test脚本进行任务运行
##############################################
class ExecuteMachine:

    def __init__(self,
                 exe_bin='python',                      # python的可执行文件的全称
                 relative_path_config='../config/',     # 运行位置到config目录的相对位置
                 relative_path_run='../run/',           # 运行位置到run目录的相对位置
                 check_verbose=True,                    # 是否开启verbose检查，默认为静默模式
                 flags={},                              # 运行flag：对于一对k,v，flag形式为: "-(k) (v)"
                 param_type_config_sep_symbol='|'):    # 使用参数型添加任务时，参数名称的层分隔符号
        self.TimeFormatter = TimeFormatter()
        self.ExecuteTaskLines = []
        self.ConfigUpdateLines = []
        self.RelativePathToConfig = relative_path_config
        self.RelativePathToRun = relative_path_run
        self.Flags = flags
        self.ExecuteBin = exe_bin
        self.CheckVerbose = check_verbose
        self.ParamTypeConfigSepSymbol = param_type_config_sep_symbol

        self.ExecuteSuccessCount = 0
        self.ExecuteFailCount = 0

        try:
            # 初始化machine时读取一个config的cache
            # 防止后续运行多个任务时config被修改导致默认值不一致的问题
            self.ConfigCache = {
                'train': loadJson(self.RelativePathToConfig + 'train.json'),
                'test': loadJson(self.RelativePathToConfig + 'test.json'),
            }
        except FileExistsError as e:
            raise FileNotFoundError(f'[ExecuteMachine] Config file not found: {e}')

    def addTask(self, task_type='train',
                structure_update_config={},
                flatten_update_config={}):
        if task_type not in ['train', 'test']:
            raise ValueError(f'[ExecuteMachine] Unsupported task type: {task_type}')

        self.ExecuteTaskLines.append(task_type)

        structure_update_config = self._parseParamTypeConfig(structure_update_config, flatten_update_config)
        self.ConfigUpdateLines.append(structure_update_config)

    def addRedoTrainTask(self, dataset, version, updated_configs=None, **kwargs):
        pm = PathManager(dataset=dataset,
                         version=version)
        # 直接读取已做过的实验的config，重新做一次
        pre_config = loadJson(joinPath(pm.doc(), 'train.json'))

        # 加载之前version的参数后，如果有需要更新的参数，则在之前参数的基础上进行更新
        if updated_configs is not None:
            self._setFields(pre_config, updated_configs)

        self.addTask('train', pre_config, **kwargs)

    def _setConfig(self, task_type, fields):
        '''
        只对给定的config的fields进行设置，没有给定的fields保持config的原值
        '''
        # 修改为读取machine初始化时加载的config的cache
        # 防止在运行过程中config被修改导致默认值在多个任务运行时不一致的问题
        if task_type in self.ConfigCache:
            conf = deepcopy(self.ConfigCache.get(task_type))
        else:
            return None
        # conf = loadJson(self.RelativePathToConfig+task_type+'.json')
        self._setFields(conf, fields)
        dumpJson(conf, self.RelativePathToConfig+task_type+'.json')

    # cur_config: 当前递归的需要被修改的参数
    # cur_fields：当前递归的参数修改值
    # force：在config和fields的key匹配不上时，是否强制将dict类型的值作为value赋给key
    def _setFields(self, cur_config, cur_fields, force=False):
        for k,v in cur_fields.items():
            if type(v) != dict:
                cur_config[k] = v
            elif k in cur_config:
                self._setFields(cur_config[k], v, force)
            elif force:
                cur_config[k] = v
            else:
                print(f"[ExecuteMachine] Can not find field ’{k}‘ in: {cur_config}, skip")

    def _checkAndWrapConfig(self, task_type, conf):
        # 基本检查：设置verbose为false
        if self.CheckVerbose:
            if task_type == 'train':
                if 'training' not in conf:
                    conf['training'] = {
                        'verbose': False
                    }
                else:
                    conf['training']['verbose'] = False

            elif task_type == 'test':
                conf['verbose'] = False

        return conf

    def _parseParamTypeConfig(self, init_update_config, params):
        parsed_params = deepcopy(init_update_config)

        for k, v in params.items():
            struct_keys = k.split(self.ParamTypeConfigSepSymbol)
            struct_val = None
            if len(struct_keys) == 0:
                raise ValueError(f'[ExecuteMachine] Can not parse params type config, len is 0 for {params}')
            elif len(struct_keys) >= 1:
                struct_val = {struct_keys[-1]: v}
                for i in reversed(range(0, len(struct_keys)-1)):
                    struct_val = {struct_keys[i]: struct_val}

            # 复用field设置方法，将每一个kv对设置写入临时config中
            self._setFields(parsed_params, struct_val, force=True)

        return parsed_params

    def _summaryTasks(self):
        train_num, test_num = 0, 0
        for task in self.ExecuteTaskLines:
            if task == 'train':
                train_num += 1
            elif task == 'test':
                test_num += 1

        print(f'[ExecuteMachine] Task Summary: \n Train task: {train_num} \n Test task: {test_num}\n\n')

    def _printTask(self, idx, task_type, task_config):
        print(f'\n\n{idx}-th task: {task_type}')
        pprint(task_config, indent=4)

    def execute(self):
        self._summaryTasks()
        start_time = time.time()

        execute_flags = ' '.join([f' -{flag_key} {flag_val}' for flag_key, flag_val in self.Flags.items()])
        for i, (task_type, task_config) in enumerate(zip(self.ExecuteTaskLines, self.ConfigUpdateLines)):
            self._printTask(i, task_type, task_config)

            run_script_path = self.RelativePathToRun + task_type + '.py'
            # 包装，检查参数
            task_config = self._checkAndWrapConfig(task_type, task_config)
            # 设置参数
            self._setConfig(task_type, task_config)

            # shell运行脚本
            code = os.system(f'{self.ExecuteBin} {run_script_path} {execute_flags}')
            if code == 0:
                self.ExecuteSuccessCount += 1
            else:
                self.ExecuteFailCount += 1

        print(f'[ExecuteMachine] All {len(self.ExecuteTaskLines)} tasks done ' +
              f'({self.ExecuteSuccessCount} success, {self.ExecuteFailCount} fail)')

        total_time = time.time() - start_time
        formatted_time_str = self.TimeFormatter.format(total_time)

        print(f"[ExecuteMachine] Total time: {formatted_time_str}")
