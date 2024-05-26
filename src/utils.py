import re
import json
import glob
import numpy as np
from typing import Any
from typing import Sequence
from typing import NamedTuple

import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import tv_tensors
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import draw_segmentation_masks
from torchvision.transforms.v2 import functional as F


ImageType = Image.Image | tv_tensors.Image
ImagesType = Sequence[ImageType] | Sequence[Sequence[ImageType]]
BoxesType = dict[str, torch.Tensor | np.ndarray | list[np.ndarray]]
AnnotatedImageType = tuple[ImageType, BoxesType]
AnnotatedImagesType = Sequence[AnnotatedImageType] | Sequence[Sequence[AnnotatedImageType]]


def plot(
    imgs: ImageType | ImagesType | AnnotatedImageType | AnnotatedImagesType, 
    to_xyxy: bool = False,
    row_title: str | None = None, 
    **imshow_kwargs
) -> None:
    if isinstance(imgs, ImageType) or (isinstance(imgs, tuple) and isinstance(imgs[0], ImageType) and isinstance(imgs[1], dict)):
        imgs = [imgs]
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)

    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            boxes = None
            masks = None
            if isinstance(img, tuple):
                img, target = img
                boxes = target.get('boxes')
                masks = target.get('masks')
                if to_xyxy:
                    boxes = torch.from_numpy(np.array([
                        [box[0], box[1], box[0]+box[2], box[1]+box[3]] for box in boxes
                    ]))
            if not isinstance(img, torch.Tensor):
                img = F.to_image(img)
            if img.dtype.is_floating_point and img.min() < 0:
                # Poor man's re-normalization for the colors to be OK-ish. This
                # is useful for images coming out of Normalize()
                img -= img.min()
                img /= img.max()

            img = F.to_dtype(img, torch.uint8, scale=True)
            if boxes is not None:
                img = draw_bounding_boxes(img, boxes, colors="yellow", width=3)
            if masks is not None:
                img = draw_segmentation_masks(img, masks.to(torch.bool), colors=["green"] * masks.shape[0], alpha=.65)

            ax = axs[row_idx, col_idx]
            ax.imshow(img.permute(1, 2, 0).numpy(), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()


class NpEncoder(json.JSONEncoder):

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def open_json(filepath: str) -> Any:
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data


def save_json(filepath: str, data: Any) -> None:
    with open(filepath, 'w') as file:
        json.dump(data, file, indent=2, sort_keys=True, cls=NpEncoder)


def get_device() -> torch.device: 
    assert torch.cuda.is_available()
    return torch.device('cuda')


class Checkpoint(NamedTuple):
    weights: str
    logs: str
    epoch: int


def find_latest_checkpoints(
    shots: int,
    seed: int,
    weights_dir: str,
    logs_dir: str,
    prefix: str = '',
) -> Checkpoint | None:
    # find existing checkpoints
    weights_fp_glob = weights_dir + f'{prefix}fsl_shots_{shots}_seed_{seed}_partial_*.pth'
    weights_fps = glob.glob(weights_fp_glob)
    logs_fp_glob = logs_dir + f'{prefix}logs_train_{shots}_{seed}_partial_*.json'
    logs_fps = glob.glob(logs_fp_glob)
    if len(weights_fps) == 0 or len(logs_fps) == 0:
        return
    # check if a pair of corresponding weight and log files exist
    weights_fp_pat = weights_dir + f'{prefix}fsl_shots_{shots}_seed_{seed}_partial_(\d+).pth'
    weights_epochs = [int(re.search(weights_fp_pat, f).groups()[0]) for f in weights_fps]
    weights_idx = np.argsort(weights_epochs)[-1]
    weights_epoch = weights_epochs[weights_idx]
    logs_fp_pat = logs_dir + f'{prefix}logs_train_{shots}_{seed}_partial_(\d+).json'
    logs_epochs = [int(re.search(logs_fp_pat, f).groups()[0]) for f in logs_fps]
    logs_idx = np.argsort(logs_epochs)[-1]
    logs_epoch = logs_epochs[logs_idx]
    if weights_epoch != logs_epoch:
        return
    # return filepaths
    epoch = weights_epoch
    weights_fp = weights_fps[weights_idx]
    logs_fp = logs_fps[logs_idx]
    return Checkpoint(weights=weights_fp, logs=logs_fp, epoch=epoch)


def update_loss_dict(
    effective_loss_dict: dict[str, float] | None,
    loss_dict_reduced: dict[str, float],
    batch_multiplier: int,
) -> dict[str, float]:
    if effective_loss_dict is None:
        result = {k: v/batch_multiplier for k, v in loss_dict_reduced.items()}
    else:
        result = {}
        for key in effective_loss_dict.items():
            result[key] = effective_loss_dict[key] + loss_dict_reduced[key] / batch_multiplier
    return result

