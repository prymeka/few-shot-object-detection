import os
import sys
import glob
import argparse

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import constants as const


def summarize_validation_logs(shots: int, prefix: str = '', no_std: bool = False, abridged: bool = False) -> None:
    log_dirs = glob.glob(const.FSL_DATA_DIR+prefix+'*_logs')
    if len(log_dirs) == 0:
        print(f"No records found for {const.FSL_DATA_DIR+prefix+'*_logs'}.")
        return
    names = [os.path.basename(d).rstrip('_logs') for d in log_dirs if glob.glob(d.rstrip('/')+f'/logs_val_{shots}_*.log')]
    names.sort()

    # extract matrics
    records = []
    for name in names:
        record = {}
        fps = glob.glob(const.FSL_DATA_DIR+name+f'_logs/logs_val_{shots}_*.log')
        for fp in fps:
            with open(fp, 'r') as f:
                skip =  True
                for line in f.readlines():
                    line = line.strip('\n')
                    if line.startswith('#'):
                        skip = True
                        header = line.strip(' #')
                        if header not in record:
                            record[header] = {}
                    if not skip:
                        key, value = line.rsplit(' ', 1)
                        if (key := key.strip()) not in record[header]:
                            record[header][key] = [float(value)]
                        else: 
                            record[header][key].append(float(value))
                    if line.startswith('IoU metric:'):
                        skip = False
        records.append(record)

    if not any(map(bool, records)):
        print('No records found.')
        return

    # aggregate metrics
    for record in records:
        for header in record:
            for metric in record[header]:
                values = record[header][metric]
                mean = round(np.mean(values), 3)
                if no_std:
                    record[header][metric] = f'{mean:.3f}'
                else:
                    std = round(np.std(values), 3)
                    record[header][metric] = f'{mean:.3f} +/- {std:.3f}'

    # print metrics
    # names = [f'B{n[3:5]} ({n[7:9]})' for n in names]
    for header in records[0]:
        print(header)
        print('#'*len(header))
        print(f'{names[0]:>86}', end='')
        if no_std:
            if len(names) > 1:
                print(f'{names[1]:>15}', end='')
            if len(names) > 2:
                for prefix in names[2:]:
                    print(f'{prefix:>21}', end='')
        else:
            if len(names) > 1:
                print(f'{names[1]:>21}', end='')
            if len(names) > 2:
                for prefix in names[2:]:
                    print(f'{prefix:>31}', end='')
        print()
        for i, metric in enumerate(records[0][header]):
            if abridged and i in (3, 4, 5, 9, 10, 11): 
                continue
            print(metric, end='')
            for j in range(len(records)):
                if j == 0:
                    # change = '          '
                    change = ''
                else: 
                    baseline = float(records[0][header][metric].split(' ', 1)[0])
                    current = float(records[j][header][metric].split(' ', 1)[0])
                    if np.isclose(baseline, 0): 
                        change = '( ---.--%)'
                    else:
                        change_direction = '+' if current >= baseline else '-'
                        numeric_change = f'{round(current / baseline, 2):.2f}'
                        change = f'({change_direction}{numeric_change:>6}%)'
                if no_std:
                    print(f'{records[j][header][metric]:>10} {change}', end='')
                else:
                    print(f'{records[j][header][metric]:>20} {change}', end='')
            print()
        print('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--shots', type=int, required=True)
    parser.add_argument('-p', '--prefix', type=str, default='')
    parser.add_argument('--no-std', action='store_true')
    parser.add_argument('-a', '--abridged', action='store_true')
    args = parser.parse_args()
    summarize_validation_logs(args.shots, args.prefix, args.no_std, args.abridged)

