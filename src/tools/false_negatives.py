import glob
from collections import defaultdict

import torch
import numpy as np
from torchvision.ops import box_iou

import constants as const
from utils import open_json
from utils import save_json


def calculate_false_negatives(
    predictions: dict[str, dict[str, torch.Tensor]],
    targets: dict[str, dict[str, torch.Tensor]],
) -> dict[str, int]:
    LOWER_BOUND = 0.1
    image_ids = predictions.keys()
    counter = {'base': {'false-negatives': 0, 'true-positives': 0}, 'novel': {'false-negatives': 0, 'true-positives': 0}}
    for id_ in image_ids: 
        gt_boxes = targets[id_]['boxes']
        dt_boxes = predictions[id_]['boxes']

        gt_labels = targets[id_]['labels'].data.numpy().astype('int')
        is_novel = max(gt_labels) > 90
        dataset = 'novel' if is_novel else 'base'
        num_true_positives = len(gt_labels)
        counter[dataset]['true-positives'] += num_true_positives

        if len(dt_boxes.shape) == 2:
            ious = box_iou(dt_boxes, gt_boxes).data.numpy()
            false_neg_idx = np.where(np.max(ious, axis=0) < LOWER_BOUND)[0]
            num_false_negatives = len(false_neg_idx)
        else:
            num_false_negatives = num_true_positives
        counter[dataset]['false-negatives'] += num_false_negatives
    return counter


def aggregate_false_negatives(shots: int) -> None:
    counter = {'base': defaultdict(list), 'novel': defaultdict(list)}
    for ft_mode in ('append', 'replace'): 
        for pred_mode in ('normal', 'cosine'):
            for aug in range(3):
                print(f'[{shots} Shot] Aggregating: {ft_mode}-{pred_mode}-{aug}...')
                stem = f'{ft_mode}_{pred_mode}_{aug}_{shots}_'
                target_fps = glob.glob(const.ERROR_DIR+stem+'*_targets.json')
                outputs_fps = glob.glob(const.ERROR_DIR+stem+'*_outputs.json')

                for tar_fp, out_fp in zip(target_fps, outputs_fps): 
                    targets = open_json(tar_fp)
                    predictions = open_json(out_fp)
                    for id_ in predictions:
                        predictions[id_] = {
                            'boxes': torch.Tensor(predictions[id_]['boxes']),
                            'labels': torch.Tensor(predictions[id_]['labels'])
                        }
                    for id_ in targets:
                        targets[id_] = {
                            'boxes': torch.Tensor(targets[id_]['boxes']),
                            'labels': torch.Tensor(targets[id_]['labels'])
                        }
                    count = calculate_false_negatives(predictions, targets)
                    frac_base = count['base']['false-negatives'] / count['base']['true-positives']
                    frac_novel = count['novel']['false-negatives'] / count['novel']['true-positives']
                    counter['base'][f'{ft_mode}-{pred_mode}-{aug}'].append(frac_base)
                    counter['novel'][f'{ft_mode}-{pred_mode}-{aug}'].append(frac_novel)
    counter['base'] = dict(counter['base'])
    counter['novel'] = dict(counter['novel'])
    save_json(const.ERROR_DIR+f'false_negatives_{shots}.json', counter)


if __name__ == '__main__':
    for shots in range(1, 4):
        aggregate_false_negatives(shots)

