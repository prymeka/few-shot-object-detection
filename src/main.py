import os
import time
import json
import shutil
import contextlib

import torch 
import torchvision

import model_utils as mu
import constants as const
from history import History
from utils import open_json
from utils import get_device
from dataset import get_datasets
from engine.engine import evaluate
from engine.engine import validate
from augmentations import get_transforms
from augmentations import ExtraAugPolicy
from engine.engine import train_one_epoch
from utils import find_latest_checkpoints
from augmentations import ExtraAugmentation

from logger_utils import get_logger
from logger_utils import LoggerWriter


torch.multiprocessing.set_sharing_strategy('file_system')
torchvision.disable_beta_transforms_warning()
torch.cuda.empty_cache()


def fine_tune(
    dir_prefix: str,
    shots: int,
    seed: int,
    num_epochs: int = 1_000,
    val_interval: int = 100,
    skip_first_val: bool = False,
    print_freq: int = 10, 
    batch_multiplier: int = 1,
    **kwargs
) -> bool:
    lr = const.LEARNING_RATE
    momentum = const.MOMENTUM
    weight_decay = const.DECAY
    batch_size = const.BATCH_SIZE
    val_batch_size = const.BATCH_SIZE
    transforms_mode = 2

    training_meta = {
        'num_epochs': num_epochs,
        'lr': lr,
        'momentum': momentum,
        'weight_decay': weight_decay,
        'batch_size': batch_size,
        'batch_multiplier': batch_multiplier,
        'val_batch_size': val_batch_size,
        'transforms_mode': transforms_mode,
        'starting_weights': kwargs['novel_weights_fp'],
        'use_cosine': kwargs['use_cosine'],
    }
    print(f'Training parameters: {json.dumps(training_meta, sort_keys=True, indent=4)}.')

    weights_dir = const.FSL_DATA_DIR + dir_prefix + 'weights/'
    os.makedirs(weights_dir, exist_ok=True)
    plots_dir = const.FSL_DATA_DIR + dir_prefix + 'plots/'
    os.makedirs(plots_dir, exist_ok=True)
    logs_dir = const.FSL_DATA_DIR + dir_prefix + 'logs/'
    os.makedirs(logs_dir, exist_ok=True)
    device = get_device()

    # construct data loaders
    data_loader = get_datasets(
        const.FSL_ANNOTATIONS_DIR+f'fsl_shots_{shots}_seed_{seed}.json', 
        root=const.DATA_DIR,
        transforms=get_transforms(apply_color_jitter=True), 
        batch_size=batch_size,
    )
    data_loader_val = get_datasets(
        const.FSL_VAL_ANNOTATIONS, 
        root=const.DATA_DIR,
        transforms=get_transforms(), 
        batch_size=val_batch_size,
    )
    data_loader_val_cppe = get_datasets(
        const.CPPE_VAL_ANNOTATIONS, 
        root=const.CPPE_VAL_IMAGES,
        transforms=get_transforms(), 
        batch_size=val_batch_size,
    )

    # construct the model
    model = mu.get_default_model()
    model = mu.extend_model_to_fsod(model, cppe_only=False, **kwargs)
    model = mu.freeze_for_fine_tuning(model, kwargs['mode'])

    # check for weights from interrupted trianing
    print('Checking for checkpoints...')
    checkpoint = find_latest_checkpoints(shots, seed, weights_dir, logs_dir)
    if checkpoint is None:
        print('No checkpoints found.')
        start_epoch = 1
        train_history = History('train')
        val_history = History('val', val_interval)
        val_history_cppe = History('val_cppe', val_interval)
    else: 
        # load model weights
        state_dict = torch.load(checkpoint.weights)
        model.load_state_dict(state_dict)
        start_epoch = checkpoint.epoch
        # load history
        past_histories = open_json(checkpoint.logs)
        train_history = History('train', starting_point=past_histories['loss']['train'])
        val_history = History('val', val_interval, starting_point=past_histories['loss']['val'])
        val_history_cppe = History('val_cppe', val_interval, starting_point=past_histories['loss']['val_cppe'])

        print(f'Found checkpoint weights. Starting at epoch {start_epoch}.')

    model = model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )
    # train loop
    scaler = torch.cuda.amp.GradScaler()
    try:
        tic = time.perf_counter()
        for epoch in range(start_epoch, num_epochs+1):
            sec = round(time.perf_counter()-tic, 2) 
            train_metric = train_one_epoch(
                model,
                optimizer,
                data_loader,
                device,
                epoch,
                print_freq=print_freq,
                scaler=scaler,
                header_prefix=f'[{sec:.2f}s] ({shots}:{seed}) ',
                extra_augmentation=ExtraAugmentation(ExtraAugPolicy.MOSAIC),
                batch_multiplier=batch_multiplier,
            )
            train_history.update(metric_logger=train_metric)
            # validation
            if (epoch == 1 or epoch % val_interval == 0) and not (skip_first_val and epoch == start_epoch):
                sec = round(time.perf_counter()-tic, 2) 
                val_metric = validate(
                    model, 
                    data_loader_val, 
                    device, 
                    epoch, 
                    print_freq=print_freq,
                    header_prefix=f'[{sec:.2f}s] ({shots}:{seed}) '
                )
                val_history.update(metric_logger=val_metric)
                sec = round(time.perf_counter()-tic, 2) 
                cppe_val_metric = validate(
                    model, 
                    data_loader_val_cppe, 
                    device, 
                    epoch, 
                    print_freq=print_freq,
                    header_prefix=f'[{sec:.2f}s] ({shots}:{seed}) '
                )
                val_history_cppe.update(metric_logger=cppe_val_metric)
    except KeyboardInterrupt:
        print('Saving gracefully...')
        torch.save(model.state_dict(), weights_dir+f'fsl_shots_{shots}_seed_{seed}_partial_{epoch}.pth')
        if epoch > 1:
            filepath = logs_dir+f'logs_train_{shots}_{seed}_partial_{epoch}.json' 
            train_history.save(filepath, [val_history, val_history_cppe], **training_meta)
            train_history.plot(plots_dir, f'fsl_{shots}_{seed}_', [val_history, val_history_cppe])
        do_eval = input('Evaluate? [y/N]: ')
        if do_eval.casefold().strip() == 'y':
            return True
        return False

    # save
    torch.save(model.state_dict(), weights_dir+f'fsl_shots_{shots}_seed_{seed}.pth')
    filepath = logs_dir+f'logs_train_{shots}_{seed}.json' 
    train_history.save(filepath, [val_history, val_history_cppe], **training_meta)
    train_history.plot(plots_dir, f'fsl_{shots}_{seed}_', [val_history, val_history_cppe])
    return True


def evaluate_fine_tuned(dir_prefix: str, shots: int, seed: int, **kwargs) -> None:
    weights_dir = const.FSL_DATA_DIR + dir_prefix + 'weights/'
    logs_dir = const.FSL_DATA_DIR + dir_prefix + 'logs/'

    device = get_device()
    # load the model
    model = mu.get_default_model()
    model = mu.extend_model_to_fsod(model, cppe_only=False, **kwargs)
    state_dict = torch.load(weights_dir+f'fsl_shots_{shots}_seed_{seed}.pth')
    model.load_state_dict(state_dict)
    model = model.to(device)
    # construct data loaders
    data_loader_both = get_datasets(
        const.FSL_VAL_ANNOTATIONS, 
        root=const.DATA_DIR,
        transforms=get_transforms(), 
        batch_size=const.BATCH_SIZE,
    )
    data_loader_coco = get_datasets(
        const.FSL_VAL_ANNOTATIONS_COCO,
        root=const.DATA_DIR,
        transforms=get_transforms(), 
        batch_size=const.BATCH_SIZE,
    )
    data_loader_cppe = get_datasets(
        const.FSL_VAL_ANNOTATIONS_CPPE,
        root=const.DATA_DIR,
        transforms=get_transforms(), 
        batch_size=const.BATCH_SIZE,
    )
    
    logger = get_logger(
        log_to_console=True,
        console_format='%(message)s',
        log_to_file=True,
        file_format='%(message)s',
        filename=logs_dir+f'logs_val_{shots}_{seed}.log'
    )
    with contextlib.redirect_stdout(LoggerWriter(logger.info)):
        c, _ = shutil.get_terminal_size()
        print_freq = -1
        header = 'BOTH'
        print('#'*(c//2-len(header)-1), header, '#'*(c//2-len(header)-1))
        evaluate(model, data_loader_both, device=device, print_freq=print_freq)
        header = 'COCO'
        print('#'*(c//2-len(header)-1), header, '#'*(c//2-len(header)-1))
        evaluate(model, data_loader_coco, device=device, print_freq=print_freq)
        header = 'CPPE-5'
        print('#'*(c//2-len(header)-1), header, '#'*(c//2-len(header)-1))
        evaluate(model, data_loader_cppe, device=device, print_freq=print_freq)

if __name__ == '__main__':
    shots, seed, num_epochs, val_interval = 1, 0, 2_000, 100

    start = time.perf_counter()
    dir_prefix = 'replace_normal_0'
    novel_weights_fp = const.CPPE_DATA_DIR + f'cw_0_weights/cppe_shots_{shots}_seed_{seed}.pth'
    kwargs = {'mode': mu.ModelExtendMode.REPLACE, 'novel_weights_fp': None, 'use_cosine': False}
    success = fine_tune(
        dir_prefix=dir_prefix,
        shots=shots,
        seed=seed,
        num_epochs=num_epochs,
        val_interval=val_interval,
        skip_first_val=False, 
        print_freq=50,
        batch_multiplier=1,
        **kwargs,
    )
    ft = time.perf_counter()

    if success:
        evaluate_fine_tuned(dir_prefix, shots, seed, **kwargs)
        ev = time.perf_counter()
        print(f'Finished evaluation in {round(ev-ft, 4)}s.')
    print(f'Finished FT in {round(ft-start, 4)}s.')

