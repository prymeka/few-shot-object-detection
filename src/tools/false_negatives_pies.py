import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import constants as const
from utils import open_json


plt.rcParams.update({'font.size': 20})


def load_false_negative_data() -> dict[str, dict]:
    data = {'base': defaultdict(dict), 'novel': defaultdict(dict)}
    for shots in range(1, 4):
        raw = open_json(const.ERROR_DIR+f'false_negatives_{shots}.json') 
        for dataset in ('base', 'novel'):
            for key, value in raw[dataset].items():
                mean = np.mean(value)*100
                std = np.std(value)*100
                data[dataset][key][f'{shots} Shots'] = (mean, std)
    data['base'] = dict(data['base'])
    data['novel'] = dict(data['novel'])
    return data


def false_negatives_pie_chart(dataset: str) -> None:
    bar_data = load_false_negative_data()
    categories = list(bar_data['base'].keys())
    categories.sort()
    bar_width = 0.3

    labels = ('3 Shots', '2 Shots', '1 Shots')
    colors = {'1 Shots': 'blue', '2 Shots': 'green', '3 Shots': 'red'}

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(24, 15))

    for i, category in enumerate(categories):
        data = bar_data[dataset][category]
        pos = i*1.1
        for label in labels:
            mean, error = data[label]
            color = colors[label]
            ax.barh(pos, mean, xerr=error, color=color, alpha=1, label=label, height=bar_width, capsize=4)
            pos += bar_width
    ax.set_title(f'False Negative Rate for {dataset.capitalize()} Dataset')
    ax.set_xlabel('Percentage of Missed Objects')
    ax.tick_params(axis='y', labelrotation=30)
    ticks_pos = np.arange(len(categories))*1.1 + bar_width
    ax.set_yticks(ticks_pos)

    tick_labels = []
    aug_labels = ('(none)', '(mos)', '(mos+col)')
    for label in categories:
        ft, pred, aug = label.split('-')
        tick_labels.append(f'{ft.capitalize()} {pred} {aug_labels[int(aug)]}')

    ax.set_yticklabels(tick_labels)
    _handles, _labels = ax.get_legend_handles_labels()
    by_label = dict(zip(_labels, _handles))
    ax.legend(by_label.values(), by_label.keys())

    # ax.set_xlim(30)

    ax.set_axisbelow(True)
    ax.grid(color='gray', linestyle='dashed')
    plt.show()


if __name__ == '__main__':
    false_negatives_pie_chart('novel')
    false_negatives_pie_chart('base')

