import os
import sys
import glob
import argparse
from collections import defaultdict

import pyperclip
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import constants as const


keys_map = {
    'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]': 'mAP',
    'Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]': 'AP50',
    'Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ]': 'AP75',
    'Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]': 'APs',
    'Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]': 'APm',
    'Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]': 'APl',
    'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]': 'AR1',
    'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]': 'AR10',
    'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]': 'AR100',
    'Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]': 'ARs',
    'Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]': 'ARm',
    'Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]': 'ARl',
}


def extract_data_from_file(filepath: str) -> dict:
    with open(filepath, 'r') as file:
        lines = file.readlines()
    data = defaultdict(dict)
    for line in lines:
        line = line.strip('\n')
        if line.startswith('#'):
            skip = True
            header = line.strip(' #')
        if not skip:
            key, value = line.rsplit(' ', 1)
            key = key.strip(' =')
            key = keys_map[key]
            data[header][key] = float(value)
        if line.startswith('IoU metric:'):
            skip = False
    return data


def extract_metric_data(dataset: str = 'CPPE-5', metric: str = 'mAP') -> pd.DataFrame:
    shots, seeds = [1, 2, 3], [0, 1, 2]
    dirs = glob.glob(const.FSL_DATA_DIR+f'*_logs/')
    all_rows = []
    for dir_ in dirs:
        name = dir_.rsplit('/')[-2].rstrip('_logs').replace('_', '-')
        row = [name]
        for shot in shots:
            values = []
            for seed in seeds: 
                file_data = extract_data_from_file(dir_+f'logs_val_{shot}_{seed}.log')
                value = file_data[dataset][metric]
                values.append(value)
            mean = round(np.mean(values)*100, 1)
            std = max(round(np.std(values)*100, 1), 0.1)
            row.append(f'{mean:.1f} \pm {std:.1f}')
        all_rows.append(row)
    columns = ['Method']+['\\text{'+str(shot)+'}' for shot in shots]
    df = pd.DataFrame(all_rows, columns=columns)
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, required=True)
    parser.add_argument('-m', '--metric', type=str, required=True)
    args = parser.parse_args()

    df = extract_metric_data(args.dataset, args.metric)
    df = df.sort_values(by='Method')
    df = df.set_index('Method', drop=True)
    result = df.to_latex()
    pyperclip.copy(result)
    print(result)

