import os
import time

import torch

import model_utils as mu
import constants as const
from history import History
from utils import get_device
from dataset import get_datasets
from engine.engine import validate
from augmentations import get_transforms
from engine.engine import train_one_epoch


def fine_tune_on_novel(dir_prefix: str, shots: int, seed: int, num_epochs: int = 1_000, val_interval: int = 100, print_freq: int = 10, mode: mu.ModelExtendMode = mu.ModelExtendMode.REPLACE) -> None:
    lr = const.LEARNING_RATE
    momentum = const.MOMENTUM
    weight_decay = const.DECAY
    batch_size = const.BATCH_SIZE
    val_batch_size = const.BATCH_SIZE
    transforms_mode = 0

    training_meta = {
        'num_epochs': num_epochs,
        'lr': lr,
        'momentum': momentum,
        'weight_decay': weight_decay,
        'batch_size': batch_size,
        'val_batch_size': val_batch_size,
        'transforms_mode': transforms_mode,
    }

    weights_dir = const.CPPE_DATA_DIR + dir_prefix + 'weights/'
    os.makedirs(weights_dir, exist_ok=True)
    plots_dir = const.CPPE_DATA_DIR + dir_prefix + 'plots/'
    os.makedirs(plots_dir, exist_ok=True)
    logs_dir = const.CPPE_DATA_DIR + dir_prefix + 'logs/'
    os.makedirs(logs_dir, exist_ok=True)
    device = get_device()
    # construct data loaders
    fn = f'cppe_shots_{shots}_seed_{seed}.json'
    data_loader = get_datasets(
        const.CPPE_ANNOTATIONS_DIR+fn, 
        root=const.CPPE_TRAIN_IMAGES,
        transforms=get_transforms(), 
        batch_size=batch_size,
    )
    data_loader_val = get_datasets(
        const.CPPE_VAL_ANNOTATIONS, 
        root=const.CPPE_VAL_IMAGES,
        transforms=get_transforms(), 
        batch_size=val_batch_size,
    )
    # construct the model
    model = mu.get_default_model()
    model = mu.extend_model_to_fsod(model, use_cosine=True, cppe_only=True, mode=mode)
    model = mu.freeze_for_fine_tuning(model, mode=mode)
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
    train_history = History('train')
    val_history = History('val', val_interval)
    scaler = torch.cuda.amp.GradScaler()

    tic = time.perf_counter()
    for epoch in range(1, num_epochs+1):
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
            extra_augmentation=None,
        )
        train_history.update(metric_logger=train_metric)
        # validation
        if epoch == 1 or epoch % val_interval == 0:
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

    # save
    torch.save(model.state_dict(), weights_dir+f'cppe_shots_{shots}_seed_{seed}.pth')
    filepath = logs_dir+f'logs_train_{shots}_{seed}.json' 
    train_history.save(filepath, [val_history], **training_meta)
    train_history.plot(plots_dir, f'cppe_{shots}_{seed}_', [val_history])


if __name__ == '__main__':
    shots = const.SHOTS
    seeds = range(const.SEED_START, const.SEED_STOP)
    num_epochs = 1_000

    tic = time.perf_counter()
    for shot in shots:
        for seed in seeds:
            fine_tune_on_novel('append_0_', shot, seed, num_epochs, val_interval=20, print_freq=10, mode=mu.ModelExtendMode.APPEND)
    print(f'Finished in {round(time.perf_counter()-tic, 2)}s.')

