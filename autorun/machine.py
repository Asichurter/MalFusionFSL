import os
from pprint import pprint

from utils.file import loadJson, dumpJson


class ExecuteMachine:

    def __init__(self,
                 exe_bin='python',
                 relative_path_config='../config/',
                 relative_path_run='../run/'):

        self.ExecuteTaskLines = []
        self.ConfigUpdateLines = []
        self.RelativePathToConfig = relative_path_config
        self.RelativePathToRun = relative_path_run
        self.Flags = {}
        self.ExecuteBin = exe_bin

    def addTask(self, task_type='train', update_config_fields={}):
        if task_type not in ['train', 'test']:
            raise ValueError(f'[ExecuteMachine] Unsupported task type: {task_type}')
        self.ExecuteTaskLines.append(task_type)
        self.ConfigUpdateLines.append(update_config_fields)

    def _setConfig(self, task_type, fields):
        conf = loadJson(self.RelativePathToConfig+task_type+'.json')
        self._setFields(conf, fields)
        dumpJson(conf, self.RelativePathToConfig+task_type+'.json')

    def _setFields(self, cur_config, cur_fields):
        for k,v in cur_fields.items():
            if type(v) != dict:
                cur_config[k] = v
            elif k in cur_config:
                self._setFields(cur_config[k], v)
            else:
                print(f"[ExecuteMachine] Can not find field {k} in {cur_config}, skip")

    def _checkAndWrapConfig(self, task_type, conf):
        # 基本检查：设置verbose为false
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

    def execute(self):
        execute_flags = ' '.join([f' -{flag_key} {flag_val}' for flag_key, flag_val in self.Flags.items()])
        for i, (task_type, task_config) in enumerate(zip(self.ExecuteTaskLines, self.ConfigUpdateLines)):
            print(f'\n\n{i}-th {task_type} task\nconfig:')
            pprint(task_config)
            run_script_path = self.RelativePathToRun + task_type + '.py'
            task_config = self._checkAndWrapConfig(task_type, task_config)
            self._setConfig(task_type, task_config)

            os.system(f'{self.ExecuteBin} {run_script_path} {execute_flags}')
        print(f'[ExecuteMachine] All {len(self.ExecuteTaskLines)} tasks done')
