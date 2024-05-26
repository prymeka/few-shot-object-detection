import sys
import math
import time
import pycocotools

import torch
from torchvision.models.detection.faster_rcnn import FasterRCNN

import engine.utils as utils
from engine.coco_eval import CocoEvaluator
from augmentations import ExtraAugmentation


def validate(
    model: FasterRCNN, 
    data_loader: torch.utils.data.DataLoader, 
    device: torch.device,
    epoch: int,
    print_freq: int,
    lr: float = 0.01,
    header_prefix: str = ''
) -> utils.MetricLogger:
    metric_logger = utils.MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = header_prefix + f'Validation: [{epoch}]'

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        with torch.no_grad():
            loss_dict = model(images, targets)
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=lr)

    return metric_logger


def train_one_epoch(
    model: FasterRCNN, 
    optimizer: torch.optim.Optimizer, 
    data_loader: torch.utils.data.DataLoader, 
    device: torch.device, 
    epoch: int, 
    print_freq: int, 
    scaler: torch.cuda.amp.grad_scaler.GradScaler | None = None,
    header_prefix: str = '',
    extra_augmentation: ExtraAugmentation | None = None,
    batch_multiplier: int = 1,
) -> utils.MetricLogger:
    model.train()
    metric_logger = utils.MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = header_prefix + f'Epoch: [{epoch}]'

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader)-1)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for i, (images, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # apply extra augmentation
        if extra_augmentation is not None:
            images, targets = extra_augmentation(images, targets)

        # forward pass
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        losses /= batch_multiplier
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()
        if not math.isfinite(loss_value):
            print(f'Loss is {loss_value}, stopping training')
            print(loss_dict_reduced)
            sys.exit(1)
        
        # backward pass
        if scaler is not None:
            scaler.scale(losses).backward()
        else:
            losses.backward()

        if (i+1) % batch_multiplier == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

            if lr_scheduler is not None:
                lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])

    return metric_logger


@torch.inference_mode()
def evaluate(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, device: torch.device, print_freq: int = 100) -> CocoEvaluator:
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device('cpu')
    model.eval()
    metric_logger = utils.MetricLogger(delimiter='  ')
    header = 'Test:'

    coco = data_loader.dataset.coco
    assert isinstance(coco, pycocotools.coco.COCO)
    coco_evaluator = CocoEvaluator(coco)

    results = {}
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = [img.to(device) for img in images]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target['image_id'].item(): output for target, output in zip(targets, outputs)}
        results.update(res)
        time.sleep(0.1)
    evaluator_time = time.time()
    coco_evaluator.update(results)
    evaluator_time = time.time() - evaluator_time
    metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('Averaged stats:', metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator

