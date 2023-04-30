from .logger import Logger
from os import path
from typing import List
from csv import DictWriter
import os
import json



class PerformanceEvalLogger(Logger):

    def __init__(self, experiment_id: str, module: str, log_dir_path: str=None, out_dir_path: str=None):
        super().__init__(experiment_id, module, log_dir_path, out_dir_path)
        self._report_path = path.join(self._output_dir, f'{experiment_id}_{module}_report.csv')

    def log_eval_report(self, reports: dict | List[dict]):
        if not isinstance(reports, List): reports = [reports]
        fieldnames = reports[0].keys()
        file_exists = os.path.isfile(self._report_path)
        with open(self._report_path, mode="a+") as f:
            csv_writer = DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
            if not file_exists:
                csv_writer.writeheader()
            csv_writer.writerows(reports)
