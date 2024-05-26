import os
from typing import Any
from pathlib import Path
from typing import Sequence
from typing import Iterable

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from utils import save_json
from engine.utils import MetricLogger


class History:

    def __init__(self, name: str, interval: int = 1, starting_point: dict[str, list[float]] | None = None) -> None:
        self.name = name
        self.interval = interval

        if starting_point is None:
            self.loss = []
            self.loss_classifier = []
            self.loss_box_reg = []
            self.loss_objectness = []
            self.loss_rpn_box_reg = []
        else:
            self.loss = starting_point['loss']
            self.loss_classifier = starting_point['loss_classifier'] 
            self.loss_box_reg = starting_point['loss_box_reg']
            self.loss_objectness = starting_point['loss_objectness']
            self.loss_rpn_box_reg = starting_point['loss_rpn_box_reg']
        self.metrics = {
            'loss': self.loss,
            'loss_classifier': self.loss_classifier,
            'loss_box_reg': self.loss_box_reg,
            'loss_objectness': self.loss_objectness,
            'loss_rpn_box_reg': self.loss_rpn_box_reg,
        }
        self._style = {'label': self.name, 'linestyle': '-', 'marker': 'o', 'markersize': 2}
        self._style_val = {'label': self.name, 'linestyle': '--', 'marker': 'o', 'markersize': 2}

    def update(
        self, 
        loss: Sequence[float] | None = None, 
        loss_classifier: Sequence[float] | None = None, 
        loss_box_reg: Sequence[float] | None = None, 
        loss_objectness: Sequence[float] | None = None, 
        loss_rpn_box_reg: Sequence[float] | None = None, 
        *,
        metric_logger: MetricLogger | None = None
    ) -> None:
        metrics = (
            loss,
            loss_classifier,
            loss_box_reg,
            loss_objectness,
            loss_rpn_box_reg,
        )
        if metric_logger is None:
            if None in metrics:
                raise ValueError(f'All metrics must be not None if metric_logger is None.')
            for metric_name, metric_value in zip(self.metrics, metrics):
                if isinstance(metric_value, torch.Tensor):
                    metric_value = metric_value.numpy()
                if isinstance(metric_value, (np.ndarray, tuple)):
                    metric_value = list(metric_value)
                if isinstance(metric_value, (float, int)):
                    metric_value = [metric_value]
                self.metrics[metric_name] += metric_value
        else:
            for metric_name in self.metrics:
                metric_value = metric_logger.__getattr__(metric_name).global_avg
                if isinstance(metric_value, torch.Tensor):
                    metric_value = metric_value.numpy()[0]
                self.metrics[metric_name].append(metric_value)

    def _get_x(self) -> Iterable[int]:
        if len(self.loss) == 0:
            return []
        elif self.interval == 1:
            return list(range(self.interval, len(self.loss)*self.interval+1, self.interval)) 
        else:
            return [1] + list(range(self.interval, len(self.loss)*self.interval, self.interval))

    def _plot_learning_curve(
        self, 
        fp: str, 
        data: list[list[int], list[float], dict]
    ) -> None:
        """
        `data` must be a list of: x-values, y-values, style arguments.
        """
        fig, ax = plt.subplots()
        ymin, ymax = 0, 0
        for x, y, style in data:
            ax.plot(x, y, **style)
            ymin = min(y) if min(y) < ymin else ymin
            ymax = max(y) if max(y) > ymax else ymax
        ax.set_title('Learning Curve (loss)')
        ax.set_xlim(min(data[0][0]), max(data[0][0]))
        ax.set_ylim(0, ymax*1.1)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        fig.legend()
        fig.savefig(fp)
        plt.close()

    def _plot_losses(
        self,
        fp: str,
        losses_data: list[list[list[int]], list[float], dict],
    ) -> None:
        """
        `losses_data` must be a list of: a list of x-values for each kind of loss, y-values, style arguments.
        """
        titles = (
            'Classifier Loss',
            'Box Reg. Loss',
            'Objectness Loss',
            'RPN Box Reg. Loss',
        )

        fig, axs = plt.subplots(2, 2, figsize=(10, 7))
        for i, ax in enumerate(axs.flatten()):
            ymin, ymax = 0, 0
            for x, ys, style in losses_data:
                y = ys[i]
                ax.plot(x, y, **style)
                ymin = min(y) if min(y) < ymin else ymin
                ymax = max(y) if max(y) > ymax else ymax
            ax.set_xlim(min(losses_data[0][0]), max(losses_data[0][0]))
            ax.set_ylim(0, ymax*1.1)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_title(titles[i])

        fig.tight_layout()
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')
        fig.savefig(fp)
        plt.close('all')

    def plot(self, dir_: str = './plots/', fn_prefix: str = '', histories: Sequence[Any] | None = None) -> None:
        plt.style.use('ggplot')
        Path(dir_).mkdir(parents=True, exist_ok=True)
        dir_ = os.path.normpath(dir_)+'/'

        if 0 in (lengths := [len(history) for history in histories]):
            print(f'Warning: empty history in histories will be ignored. Given {lengths = }.')
        histories = [history for history in histories if len(history) > 0]

        # learning curve
        x = self._get_x()
        y = self.loss
        data = [[x, y, self._style]]
        if histories is not None: 
            for history in histories:
                data.append([
                    history._get_x(),
                    history.loss,
                    history._style_val,
                ])
        fp = dir_+fn_prefix+'learning_curve.png'
        self._plot_learning_curve(fp, data)
        # learning curve close up
        x = self._get_x()
        y = self.loss
        n = len(x) // 2
        data = [[x[n:], y[n:], self._style]]
        if histories is not None: 
            for history in histories:
                x_val = history._get_x()
                n_val = len(x_val) // 2
                data.append([
                    x_val[n_val:],
                    history.loss[n_val:],
                    history._style_val,
                ])
        fp_closeup = dir_+fn_prefix+'learning_curve_closeup.png'
        self._plot_learning_curve(fp_closeup, data)
        # trianing loss only learning curve close up
        fp_closeup = dir_+fn_prefix+'learning_curve_closeup_train.png'
        self._plot_learning_curve(fp_closeup, data[:1])

        # losses graph
        x = self._get_x()
        ys = [self.loss_classifier, self.loss_box_reg, self.loss_objectness, self.loss_rpn_box_reg]
        data = [[x, ys, self._style]]
        if histories is not None:
            for history in histories:
                data.append([
                    history._get_x(),
                    [history.loss_classifier, history.loss_box_reg, history.loss_objectness, history.loss_rpn_box_reg],
                    history._style_val,
                ])
        fp = dir_+fn_prefix+'losses.png'
        self._plot_losses(fp, data)
        # losses graph close up
        x = self._get_x()
        ys = [self.loss_classifier[n:], self.loss_box_reg[n:], self.loss_objectness[n:], self.loss_rpn_box_reg[n:]]
        data = [[x[n:], ys, self._style]]
        if histories is not None:
            for history in histories:
                x_val = history._get_x()
                n_val = len(x_val) // 2
                data.append([
                    x_val[n_val:],
                    [history.loss_classifier[n_val:], history.loss_box_reg[n_val:], history.loss_objectness[n_val:], history.loss_rpn_box_reg[n_val:]],
                    history._style_val,
                ])
        fp_closeup = dir_+fn_prefix+'losses_closeup.png'
        self._plot_losses(fp_closeup, data)
        # training loss only losses graph close up
        fp_closeup = dir_+fn_prefix+'losses_closeup_train.png'
        self._plot_losses(fp_closeup, data[:1])

    def save(self, filepath: str, histories: Sequence[Any] | None = None, **extra_kwargs) -> None: 
        data = extra_kwargs

        losses = {self.name: self.metrics}
        if histories is not None:
            for history in histories:
                losses[history.name] = history.metrics
                data[f'{history.name}_interval'] = history.interval
        data['loss'] = losses
        save_json(filepath, data)

    def __str__(self) -> str:
        n = len(self)
        s = f'\n    name={self.name},\n    length={n},\n'
        if n > 0:
            for key, value in self.metrics.items():
                s += ' '*4 + f'{key}={np.mean(value)},\n'
        return f'History({s})'

    def __len__(self) -> int:
        return len(self.loss)

