import os
import sys
from typing import Any
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from more_itertools import unzip

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import constants as const
from utils import open_json


plt.rcParams.update({'font.size': 16})


def autopct_format(labels: list[str], totals: list[int]) -> Any:
    labels.reverse()
    totals.reverse()
    i = 0
    def my_format(pct: float) -> str:
        nonlocal i
        label = labels[i]
        i += 1
        return f'{label}\n{pct:.1f}%'
    return my_format


def main_pie_chart(ft_mode: str, dataset: str, aug: int = 0) -> None:
    fig, axs = plt.subplots(2, 3, figsize=(16, 12))
    aug_str = ('no aug.', 'mosaic', 'mos+col')[aug]
    fig.suptitle(f'{ft_mode.capitalize()} Approach ({aug_str}) on {dataset.capitalize()} Dataset')
    for i, mode in enumerate(('normal', 'cosine')):
        for shot in range(1, 4):
            counter = Counter()
            for seed in range(3):
                fn = f'{ft_mode}_{mode}_{aug}_{shot}_{seed}_errors.json'
                data = open_json(const.ERROR_DIR + fn)
                data = data[dataset]
                for key in data['classification']:
                    data['classification'][key] = sum(data['classification'][key].values())
                data['classification'] = sum(data['classification'].values())
                _ = data.pop('unknown', None)
                _ = data.pop('true', None)
                counter.update(data)
            labels, sizes = unzip(list(counter.most_common()))
            labels, sizes = list(labels), list(sizes)
            sort_idx = np.argsort(labels)
            labels, sizes = list(np.array(labels)[sort_idx]), list(np.array(sizes)[sort_idx])
            my_fmt = autopct_format(labels, sizes)
            axs[i][shot-1].pie(
                sizes,
                autopct=my_fmt,
                pctdistance=0.6, 
                labeldistance=1.1,
            )
            mode_str = {'cosine': 'cosine', 'normal': 'dot-product'}[mode]
            axs[i][shot-1].set_title(f'{shot} Shot ({mode_str})')
    fig.tight_layout()
    plt.show()


def pie_chart_per_aug(ft_mode: str, pred_mode: str, dataset: str) -> None:
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    pred_str = {'cosine': 'cosine', 'normal': 'dot-product'}[pred_mode]
    fig.suptitle(f'{ft_mode.capitalize()} ({pred_str}) on {dataset.capitalize()} Dataset')
    for shot in range(1, 4):
        for i, aug in enumerate(('no aug.', 'mosaic', 'mos+col')):
            counter = Counter()
            for seed in range(3):
                fn = f'{ft_mode}_{pred_mode}_{i}_{shot}_{seed}_errors.json'
                data = open_json(const.ERROR_DIR + fn)
                data = data[dataset]
                for key in data['classification']:
                    data['classification'][key] = sum(data['classification'][key].values())
                data['classification'] = sum(data['classification'].values())
                _ = data.pop('unknown', None)
                _ = data.pop('true', None)
                counter.update(data)
            labels, sizes = unzip(list(counter.most_common()))
            labels, sizes = list(labels), list(sizes)
            sort_idx = np.argsort(labels)
            labels, sizes = list(np.array(labels)[sort_idx]), list(np.array(sizes)[sort_idx])
            my_fmt = autopct_format(labels, sizes)
            axs[shot-1][i].pie(
                sizes,
                # labels=labels,
                autopct=my_fmt,
                pctdistance=0.6, 
                labeldistance=1.1,
            )
            axs[shot-1][i].set_title(f'{shot} Shot {aug}')
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    # main_pie_chart('replace', 'novel', 2)
    # main_pie_chart('append', 'novel', 2)
    pie_chart_per_aug('append', 'normal', 'novel')

