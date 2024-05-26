import os
import gc
import sys
import time
import argparse
import resource
from collections import defaultdict

import torch
import torchvision
import numpy as np
from torchvision.ops import box_iou

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model_utils as mu
import constants as const
from utils import open_json
from utils import save_json
from utils import get_device
from dataset import get_datasets
from augmentations import get_transforms


rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

torch.multiprocessing.set_sharing_strategy('file_system')
torchvision.disable_beta_transforms_warning()
torch.cuda.empty_cache()


@torch.inference_mode()
def evaluate(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, device: torch.device) -> dict:
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device('cpu')
    model.eval()
    all_res = {}
    for images, targets in data_loader:
        images = [img.to(device) for img in images]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {
            target['image_id'].item(): {'target': target, 'output': output}
            for target, output in zip(targets, outputs)
        }
        all_res.update(res)
        del images, targets
        gc.collect()
    torch.set_num_threads(n_threads)
    return all_res


def results_to_json(results: dict, targets_filepath: str, outputs_filepath: str) -> None:
    json_outputs = {}
    json_targets = {}
    for image_id, result in results.items():
        target = result['target']
        json_targets[str(image_id)] = {
            'boxes': torch.Tensor(target['boxes']).data.numpy().tolist(),
            'labels': target['labels'].data.numpy().astype('int').tolist(),
        }
        output = result['output']
        json_outputs[str(image_id)] = {
            'boxes': torch.Tensor(output['boxes']).data.numpy().tolist(),
            'labels': output['labels'].data.numpy().astype('int').tolist(),
            'scores': output['scores'].data.numpy().tolist(),
        }
    save_json(targets_filepath, json_targets)
    save_json(outputs_filepath, json_outputs) 


def to_tensors(values: dict) -> dict:
    for id_, dict_ in values.items():
        for key, value in dict_.items():
            values[id_][key] = torch.Tensor(value)
            if key == 'boxes':
                values[id_][key] = values[id_][key].reshape(-1, 4)
    return values


def calculate_false_positives(
    predictions: dict[str, dict[str, torch.Tensor]],
    targets: dict[str, dict[str, torch.Tensor]],
) -> dict[str, int]:
    LOWER_BOUND, UPPER_BOUND = 0.1, 0.5
    image_ids = predictions.keys()

    counter = [
        # base
        {
            'background': 0,
            'localisation': 0,
            'other': 0,
            'classification': defaultdict(lambda: defaultdict(dict)),
            'true': 0,
            'unknown': 0,
        },
        # novel
        {
            'background': 0,
            'localisation': 0,
            'other': 0,
            'classification': defaultdict(lambda: defaultdict(dict)),
            'true': 0,
            'unknown': 0,
        },
    ]

    for id_ in image_ids: 
        gt_boxes = targets[id_]['boxes']
        dt_boxes = predictions[id_]['boxes']
        ious = box_iou(dt_boxes, gt_boxes).data.numpy()
        gt_labels = targets[id_]['labels'].data.numpy().astype('int')
        dt_labels = targets[id_]['labels'].data.numpy().astype('int')
        for iou, pred in zip(ious, dt_labels):
            max_iou_idx = np.argmax(iou)
            max_iou = iou[max_iou_idx]
            true = int(gt_labels[max_iou_idx])
            pred = int(pred)
            is_novel = int(true > const.COCO_NUM_CATEGORIES)
            if max_iou < LOWER_BOUND:
                # background confusion
                counter[is_novel]['background'] += 1
            elif LOWER_BOUND <= max_iou <= UPPER_BOUND and pred == true:
                # localisation error
                counter[is_novel]['localisation'] += 1
            elif LOWER_BOUND <= max_iou <= UPPER_BOUND and pred != true:
                # confusion with other 
                counter[is_novel]['other'] += 1
            elif UPPER_BOUND < max_iou:
                if pred == true:
                    # true positive
                    counter[is_novel]['true'] += 1
                else:
                    # classification error
                    if pred in counter[is_novel]['classification'][true]:
                        counter[is_novel]['classification'][true][pred] += 1
                    else:
                        counter[is_novel]['classification'][true][pred] = 1
            else:
                counter[is_novel]['unknown'] += 1
    counter = {'base': counter[0], 'novel': counter[1]}
    return counter


def run_error_analysis(dir_prefix: str, shots: int, seed: int) -> None:
    os.makedirs(const.ERROR_DIR, exist_ok=True)

    mode = mu.ModelExtendMode.APPEND if 'append' in dir_prefix else mu.ModelExtendMode.REPLACE
    use_cosine = 'normal' not in dir_prefix
    print(f'Mode = {mode.value}. Use Cosine = {use_cosine}.')
    device = get_device()
    model = mu.get_default_model()
    model = mu.extend_model_to_fsod(
        model=model, 
        novel_weights_fp=None,
        use_cosine=use_cosine,
        cppe_only=False,
        mode=mode,
    )
    weights_dir = const.FSL_DATA_DIR + dir_prefix + 'weights/'
    state_dict = torch.load(weights_dir+f'fsl_shots_{shots}_seed_{seed}.pth')
    model.load_state_dict(state_dict)
    model = model.to(device)

    data_loader = get_datasets(
        const.FSL_VAL_ANNOTATIONS, 
        root=const.DATA_DIR,
        transforms=get_transforms(), 
        batch_size=8,
    )

    targets_filepath = const.ERROR_DIR + dir_prefix + f'{shots}_{seed}_targets.json'
    outputs_filepath = const.ERROR_DIR + dir_prefix + f'{shots}_{seed}_outputs.json' 
    print('Evaluating...')
    r = evaluate(model, data_loader, device)
    results_to_json(
        results=r, 
        targets_filepath=targets_filepath,
        outputs_filepath=outputs_filepath,
    )

    print('Agregating...')
    errors_filepath = const.ERROR_DIR + dir_prefix + f'{shots}_{seed}_errors.json'
    targets = to_tensors(open_json(targets_filepath))
    outputs = to_tensors(open_json(outputs_filepath))
    counter = calculate_false_positives(outputs, targets)
    save_json(errors_filepath, counter)
    print(f'Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prefix', type=str, required=True)
    parser.add_argument('-s', '--shots', type=int, required=True)
    parser.add_argument('-sd', '--seed', type=int, required=False, default=None)
    args = parser.parse_args()
    if args.seed is None:
        seeds = [0, 1, 2]
        for seed in seeds:
            run_error_analysis(args.prefix, args.shots, seed)
    else: 
        run_error_analysis(args.prefix, args.shots, args.seed)

