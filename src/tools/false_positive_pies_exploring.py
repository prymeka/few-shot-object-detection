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


def autopct_format(labels: list[str], totals: list[int]) -> Any:
    labels.reverse()
    totals.reverse()
    i = 0
    def my_format(pct: float) -> str:
        nonlocal i
        label = labels[i]
        total = totals[i]
        i += 1
        return f'{label}\n{pct:.1f}%\n({total//3})'
    return my_format


def pie_chart_per_init(dataset: str, aug: int) -> None:
    aug_str = ('no aug.', 'mosaic', 'mos+col')[aug]
    fig, axs = plt.subplots(2, 3, figsize=(22, 14))
    fig.suptitle(f'Weight Init. Comparison for {dataset.capitalize()} Dataset (Aug.: {aug_str})')
    for i, mode in enumerate(('cosine', 'co_imp')):
        for shot in range(1, 4):
            counter = Counter()
            for seed in range(3):
                fn = f'replace_{mode}_{aug}_{shot}_{seed}_errors.json'
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
                # labels=labels,
                autopct=my_fmt,
                pctdistance=0.6, 
                labeldistance=1.1,
            )
            name = 'Imprinting' if 'imp' in mode else 'Random'
            axs[i][shot-1].set_title(f'{shot} Shot {name} ({sum(sizes)})')
    fig.tight_layout()
    plt.show()


def pie_chart_per_seed(ft_mode: str, pred_mode: str, shot: int, dataset: str) -> None:
    fig, axs = plt.subplots(3, 3, figsize=(22, 24))
    fig.suptitle(f'{shot} Shot {pred_mode.capitalize()} ({ft_mode.capitalize()})')
    for i, aug in enumerate(('no aug.', 'mosaic', 'mos+col')):
        for seed in range(3):
            fn = f'{ft_mode}_{pred_mode}_{i}_3_{seed}_errors.json'
            data = open_json(const.ERROR_DIR + fn)
            data = data[dataset]
            for key in data['classification']:
                data['classification'][key] = sum(data['classification'][key].values())
            data['classification'] = sum(data['classification'].values())
            _ = data.pop('unknown', None)
            _ = data.pop('true', None)
            counter = Counter(data)
            labels, sizes = unzip(list(counter.most_common()))
            labels, sizes = list(labels), list(sizes)
            sort_idx = np.argsort(labels)
            labels, sizes = list(np.array(labels)[sort_idx]), list(np.array(sizes)[sort_idx])
            my_fmt = autopct_format(labels, sizes)
            axs[i][seed].pie(
                sizes,
                # labels=labels,
                autopct=my_fmt,
                pctdistance=0.6, 
                labeldistance=1.1,
            )
            axs[i][seed].set_title(f'{aug} ({sum(sizes)}) Seed {seed}')
    fig.tight_layout()
    plt.show()


def pie_chart_per_cosine(ft_mode: str, dataset: str, aug: int = 0) -> None:
    fig, axs = plt.subplots(2, 3, figsize=(16, 10))
    aug_str = ('no aug.', 'mosaic', 'mos+col')[aug]
    fig.suptitle(f'{ft_mode.capitalize()} (Aug.: {aug_str}) on {dataset.capitalize()} Dataset')
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
                # labels=labels,
                autopct=my_fmt,
                pctdistance=0.6, 
                labeldistance=1.1,
            )
            mode_str = {'cosine': 'Cosinse', 'normal': 'Dot-product'}[mode]
            axs[i][shot-1].set_title(f'{shot} Shot {mode_str} ({sum(sizes)})')
    fig.tight_layout()
    plt.show()


def pie_chart_per_mode(dataset: str, aug: int = 0) -> None:
    fig, axs = plt.subplots(4, 3, figsize=(22, 24))
    aug_str = ('no aug.', 'mosaic', 'mos+col')[aug]
    fig.suptitle(f'Comparison of Fine-Tuning Method (Aug:. {aug_str}) on {dataset.capitalize()} Dataset')
    for i, ft_mode in enumerate(('replace', 'append')):
        for j, pred_mode in enumerate(('normal', 'cosine')):
            for shot in range(1, 4):
                counter = Counter()
                for seed in range(3):
                    fn = f'{ft_mode}_{pred_mode}_{aug}_{shot}_{seed}_errors.json'
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
                axs[i*2+j][shot-1].pie(
                    sizes,
                    # labels=labels,
                    autopct=my_fmt,
                    pctdistance=0.6, 
                    labeldistance=1.1,
                )
                axs[i*2+j][shot-1].set_title(f'{shot} Shot {ft_mode.capitalize()} {pred_mode.capitalize()} ({sum(sizes)})')
    fig.tight_layout()
    plt.show()


def pie_chart_per_aug(ft_mode: str, pred_mode: str, dataset: str) -> None:
    fig, axs = plt.subplots(3, 3, figsize=(22, 24))
    fig.suptitle(f'{pred_mode.capitalize()} ({ft_mode.capitalize()}) on {dataset.capitalize()} Dataset')
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
            axs[shot-1][i].set_title(f'{shot} Shot {aug} ({sum(sizes)})')
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    # pie_chart_per_seed('append', 'cosine', 1, 'base')
    # pie_chart_per_init('novel', aug=0)
    # pie_chart_per_cosine('replace', 'novel', 0)
    pie_chart_per_mode('base', 0)
    pie_chart_per_mode('base', 1)
    pie_chart_per_mode('base', 2)
    pie_chart_per_mode('novel', 0)
    pie_chart_per_mode('novel', 1)
    pie_chart_per_mode('novel', 2)
    # pie_chart_per_aug('replace', 'normal', 'novel')

