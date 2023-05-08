from pathlib import Path
import pandas as pd



def load_drift_history(exp_path: str):
    rootdir = Path(exp_path)
    dir_list = [f for f in rootdir.resolve().glob('*') if not f.is_file() and ('drift_eval' in f.name or '__' in f.name)]
    res = {}
    for dir_path in dir_list:
        drift_indices = [f.name.split('.')[0] for f in Path(dir_path).resolve().glob('*') if 'html' not in f.name]
        if '__' in dir_path.name:
            name = f"feat{dir_path.name.split('__')[1]}"
        else:
            name = dir_path.name.split('_eval')[0]
        res.update({name : drift_indices})
    return res


def get_metrics_vals(exp_path: str):
    files = [f for f in Path(exp_path).resolve().glob('*') if f.is_file() and 'report' in f.name]
    if len(files) == 0: return None
    metrics_report = files[0]
    res = pd.read_csv(metrics_report)
    return res
    