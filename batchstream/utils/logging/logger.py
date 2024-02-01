import logging
from os import path
import os
from typing import List
import json
from .base.logger_base import ILogger



class Logger(ILogger):

    def __init__(self, experiment_id: str, module: str, log_dir_path: str=None, out_dir_path: str=None):
        self._experiment_id = experiment_id
        self._module = module
        self._log_dir: str = log_dir_path if log_dir_path != None else path.join('./log', self._experiment_id)
        self._output_dir: str = out_dir_path if out_dir_path != None else path.join('./out', self._experiment_id)
        self._check_dirs()
        self._logger: logging.Logger = self._set_up_logger()

    def _check_dirs(self):
        if not path.exists(self._log_dir):
            os.makedirs(self._log_dir)
        if not path.exists(self._output_dir):
            os.makedirs(self._output_dir)

    def _set_up_logger(self):
        log_file_name = path.join(self._log_dir, f'{self._experiment_id}_{self._module}.log')
        handler = logging.FileHandler(log_file_name)
        formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s:%(message)s')
        handler.setFormatter(formatter)
        specified_logger = logging.getLogger(name=f'{self._experiment_id}_{self._module}')
        specified_logger.setLevel(logging.DEBUG)
        specified_logger.addHandler(handler)
        return specified_logger
        
    def log_exception(self, e: Exception):
        self._logger.exception(e)

    def log_info(self, msg: str):
        self._logger.info(msg)

    def log_warn(self, warning: str):
        self._logger.warn(warning)

    def log_dict_as_json(self, d: dict):
        json_file_name = path.join(self._output_dir, f'{self._experiment_id}_{self._module}.json')
        with open(json_file_name, 'a') as f:
            default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
            f.write(json.dumps(d, default=default))

    def log_time(self, start_time, end_time, name='time'):
        t = end_time - start_time
        d = {'time [s]': t}
        file_path = path.join(self._output_dir, f'{name}.json')
        with open(file_path, 'a') as f:
            default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
            f.write(json.dumps(d, default=default))

    def log_model_history(self, i: int, type: str="replacement"):
        file_path = path.join(self._output_dir, f'model_{type}_history.csv')
        with open(file_path, mode="a+") as f:        
            f.write(f'{i}\n')

    def append_to_txt(self, line: str, file_name: str):
        file_path =path.join(self._output_dir, file_name)
        with open(file_path, mode="a+") as f:        
            f.write(f'{line}\n')
            