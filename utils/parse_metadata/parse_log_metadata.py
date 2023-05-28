import json
import re
import os
import pandas as pd



def get_drift_detectors_metadata(drift_detectors):
    if drift_detectors is None: return []
    drift_handlers_metadata = []
    for d_i in drift_detectors:
        for d_i_step in d_i['monitor']['test_steps']:
            d_i_monitor = list(d_i_step.values())[0]
            if d_i_monitor['type'] == 'EvidentlyMonitoringStep':
                d_i_meta = {'min_instances': d_i_monitor['min_instances'],
                                    'name': d_i_monitor['name'],
                                    'clock': d_i_monitor['clock'],
                                    'min_instances': d_i_monitor['min_instances'],
                                    'type': 'Evidently'}
                if 'evidently_test_suite__tests' in d_i_monitor:
                    for test in d_i_monitor['evidently_test_suite__tests']:
                        if 'stattest_threshold' in test:
                            name = ''
                            if 'test_name' in test: name = test['test_name']
                            elif 'preset_name' in test: name = test['preset_name']
                            d_i_meta.update(
                                {
                                    'stattest_threshold': test['stattest_threshold'],
                                    'name': name
                                }
                            )
                drift_handlers_metadata.append(d_i_meta)            
    return drift_handlers_metadata

def get_batch_model_metadata(batch_model):
    if batch_model['type'] == 'SklearnEstimator':
        batch_alg = [x[0] for x in batch_model['sklearn_estimator']['steps'] if x[0] in ['rf']][0]
    return batch_alg

def get_monitors_info_text(handlers, h_type):
    info = ''
    if len(handlers) >= 1:
        info += f'{h_type} monitors:\n'
        for h in handlers:
            info += f"\t{h['type']} {h['name']} (clock: {h['clock']}, n_min: {h['min_instances']}"
            if 'stattest_threshold' in h:
                info += f", stattest_th: {h['stattest_threshold']}"
            if 'name' in h:
                n = h['name'].replace("'>", "")
                info += f", name: {n}"
            info += ")\n"
    return info
   

def print_info(out_dir):
    file_list = [f for f in out_dir.resolve().glob('*') if f.is_file() and 'metadata' in f.name]
    if len(file_list) == 0:
        return
    with open(file_list[0], 'r') as f:
        d = json.load(f)
    stream_pipeline = d['stream_pipeline']
    if stream_pipeline['type'] == 'BatchPipeline':
        first_fit = stream_pipeline['min_samples_first_fit']
        n_retrain = stream_pipeline['min_samples_retrain']
        batch_model = stream_pipeline['batch_model']
        batch_alg = get_batch_model_metadata(batch_model)
        i_handlers = get_drift_detectors_metadata(stream_pipeline['input_drift_detector'])
        o_handlers = get_drift_detectors_metadata(stream_pipeline['output_drift_detector'])
    info = f'BatchPipeline ({batch_alg}) - n_min_retrain: {n_retrain}, n_first_fit: {first_fit}\n'
    info += get_monitors_info_text(i_handlers, 'Input')
    info += get_monitors_info_text(o_handlers, 'Output')
    print(info)
    return info

def extract_parameters_from_json(out_dir):
    file_list = [f for f in out_dir.resolve().glob('*') if f.is_file() and 'metadata' in f.name]
    if len(file_list) == 0:
        return {}
    parameters = {}
    with open(file_list[0], 'r') as f:
        data = json.load(f)

        # Extract the parameters from the JSON data
        parameters['clock'] = extract_parameter(data, 'clock')
        parameters['delta'] = extract_parameter(data, 'delta')
        parameters['min_window_length'] = extract_parameter(data, 'min_window_length')

    return parameters

def extract_parameter(data, parameter_name):
    occurrences = []

    def find_occurrences(data, parameter_name):
        if isinstance(data, dict):
            for key, value in data.items():
                if key == parameter_name:
                    occurrences.append(value)
                elif isinstance(value, (dict, list)):
                    find_occurrences(value, parameter_name)
        elif isinstance(data, list):
            for item in data:
                find_occurrences(item, parameter_name)

    find_occurrences(data, parameter_name)
    return occurrences

def extract_values(string):
    pattern = r"clock: (\d+),.*stattest_th: ([\d.]+)"

    match = re.search(pattern, string)
    if match:
        clock = int(match.group(1))
        stattest_threshold = float(match.group(2))
        return clock, stattest_threshold
    else:
        return 5000, 0.05
    
def get_replacement_hist(d):
    path = os.path.join(str(d), 'model_replacement_history.csv')
    if os.path.exists(path):
        return list(pd.read_csv(path, header=None)[0].values)
