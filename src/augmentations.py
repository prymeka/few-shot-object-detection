import enum
from typing import Any
from abc import ABCMeta
from abc import abstractmethod

import torch
import numpy as np
from more_itertools import grouper
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms import AutoAugment
from torchvision.transforms.v2 import functional as F


SMALL_NUMBER = 1e-3
ImagesType = tuple[tv_tensors.Image]
TargetsType = tuple[dict[str, tv_tensors.BoundingBoxes | Any]]


class MosaicSubTransform(torch.nn.Module, metaclass=ABCMeta):
    """
    Batch transform meant to operate on batches of size of multiple of 4.
    """
    
    @abstractmethod
    def forward(self, images: ImagesType, targets: TargetsType) -> tuple[ImagesType, TargetsType]:
        pass


class SquarePad(MosaicSubTransform):

    def __init__(self, centre_pad: bool = False, fill_value: int = 114) -> None:
        """
        Pad images to square. If batch size is not divisible by 4, the remainder of images
        will not be transformed.

        Parameters
        ----------
        centre_pad: bool
            If `centre_pad` is `False`, images will be padded "on the outside", i.e., 
            such that no padding will be around the centre of mosaic. Otherwise, padding
            will be equal on all sides of individual images. Defaults to `False`.
        fill_value: int 
            Value to pad with. Defaults to `114` following:
            https://github.com/ultralytics/yolov5/blob/master/utils/dataloaders.py#L843
        """
        super().__init__()
        self.centre_pad = centre_pad 
        self.fill_value = fill_value

    def forward(self, images: ImagesType, targets: TargetsType) -> tuple[ImagesType, TargetsType]:
        images, targets = list(images), list(targets)
        for batch_idx in grouper(range(len(images)), n=4, incomplete='ignore'):
            for i, pos in zip(batch_idx, ['tl', 'tr', 'bl', 'br']):
                _, h, w = images[i].shape 
                if h == w:
                    continue
                max_wh = np.max([w, h]) 
                p_top, p_left = [(max_wh-s)//2 for s in (h, w)]
                p_bottom, p_right = [max_wh-(s+pad) for s, pad in zip((h, w), (p_top, p_left))]
                # padding values: (left, top, right, bottom)
                if self.centre_pad:
                    padding = (p_left, p_top, p_right, p_bottom)
                    targets[i]['boxes'][:, [0, 2]] += padding[0]
                    targets[i]['boxes'][:, [1, 3]] += padding[1]
                else:
                    if pos == 'tl':
                        padding = (p_left+p_right, p_top+p_bottom, 0, 0)
                        targets[i]['boxes'][:, [0, 2]] += padding[0]
                        targets[i]['boxes'][:, [1, 3]] += padding[1]
                    elif pos == 'tr':
                        padding = (0, p_top+p_bottom, p_left+p_right, 0)
                        targets[i]['boxes'][:, [1, 3]] += padding[1]
                    elif pos == 'bl':
                        padding = (p_left+p_right, 0, 0, p_top+p_bottom)
                        targets[i]['boxes'][:, [0, 2]] += padding[0]
                    elif pos == 'br':
                        padding = (0, 0, p_left+p_right, p_top+p_bottom)
                images[i] = F.pad(images[i], padding, self.fill_value, 'constant')
             
        return tuple(images), tuple(targets)


class Resize(MosaicSubTransform):

    def __init__(self, max_size: int) -> None:
        """
        Rescale an image so that maximum side is equal to `max_size`, keeping the aspect ratio of the initial image.
        """
        super().__init__()
        self.max_size = max_size 
    
    def forward(self, images: ImagesType, targets: TargetsType) -> tuple[ImagesType, TargetsType]:
        images, targets = list(images), list(targets)
        for i, image in enumerate(images):
            _, h, w = image.shape 
            aspect_ratio = np.min([w, h]) / np.max([w, h])
            small_side_size = int(self.max_size * aspect_ratio)
            # if int is passed to F.resize, then the small side will be matched to it
            images[i] = F.resize(images[i], small_side_size, antialias=True)
            # adjust bounding boxes and area
            resize_factor = self.max_size / np.max([w, h])
            targets[i]['boxes'] = targets[i]['boxes'].to(torch.float32)
            targets[i]['boxes'] *= resize_factor
            targets[i]['area'] = targets[i]['area'].to(torch.float32)
            targets[i]['area'] *= resize_factor
             
        return tuple(images), tuple(targets)


class RandomCrop(MosaicSubTransform):

    def __init__(
        self,
        output_size: int | tuple[int, int] | None,
        min_visibility: float | None = None,
    ) -> None:
        """
        Randomly crop the image to size.

        Parameters
        ----------
        output_size: int | tuple[int, int] | None
            The width and height of the cropped image. If an `int` is passed 
            the output will be a square. The `output_size` can be `None`, but then 
            the output size must be passed as the 3rd argument in the `forward` call.
        min_visibility: float, optional
            If `min_visibility` is a `float`, then bounding boxes with 
            `[area after crop] / [area before crop] < min_visibility` will be removed. 
            If `min_visibility <= 0.0` or `None`, then no bounding boxes will be 
            removed save for those completely outside of the cropped image.
        """
        super().__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = output_size
        self.min_visibility = min_visibility

    def forward(
        self,
        images: ImagesType,
        targets: TargetsType,
        output_size: int | tuple[int, int] | None = None,
    ) -> tuple[ImagesType, TargetsType]:
        assert self.output_size is not None or output_size is not None, f'Given {self.output_size = } and {output_size = }.'

        # override output size if provided
        output_size = self.output_size if output_size is None else output_size
        if isinstance(output_size, int):
            output_size = (output_size, output_size)

        images, targets = list(images), list(targets)
        for i, (image, target) in enumerate(zip(images, targets)):
            # find cropping coordinates
            _, h, w = image.shape 
            if w == output_size[0] and h == output_size[1]:
                continue
            x = np.random.randint(0, w-output_size[0]) if w-output_size[0] > 0 else 0
            y = np.random.randint(0, h-output_size[1]) if h-output_size[1] > 0 else 0
            # crop the image
            images[i] = image[:, y:y+output_size[1], x:x+output_size[0]]
            # adjust targets
            new_target = self._adjust_target(target, output_size, x, y)
            targets[i] = new_target

        return tuple(images), tuple(targets)

    def _adjust_target(
        self, 
        target: dict[str, tv_tensors.BoundingBoxes | Any],
        output_size: tuple[int, int],
        x: int, 
        y: int,
    ) -> dict[str, tv_tensors.BoundingBoxes | Any]:
        min_visibility = SMALL_NUMBER if self.min_visibility is None else max(0.0, self.min_visibility)
        target['boxes'] = target['boxes'].to(torch.float32)
        target['boxes'][:, [0, 2]] -= x
        target['boxes'][:, [1, 3]] -= y
        new_bboxes = []
        # loop one bounding box at a time
        keys = ['boxes', 'labels', 'iscrowd', 'area']
        for values in zip(*[target[key] for key in keys]):
            bboxes = {key: value for key, value in zip(keys, values)}
            old_area = (bboxes['boxes'][2]-bboxes['boxes'][0])*(bboxes['boxes'][3]-bboxes['boxes'][1])
            # adjust the boxes to be contained within the image
            bboxes['boxes'] = torch.where(bboxes['boxes'] < 0, 0, bboxes['boxes'])
            bboxes['boxes'][[0, 2]] = torch.where(
                bboxes['boxes'][[0, 2]] > output_size[0],
                float(output_size[0]),
                bboxes['boxes'][[0, 2]]
            )
            bboxes['boxes'][[1, 3]] = torch.where(
                bboxes['boxes'][[1, 3]] > output_size[1],
                float(output_size[1]),
                bboxes['boxes'][[1, 3]]
            )
            if bboxes['boxes'][2] <= bboxes['boxes'][0] or bboxes['boxes'][3] <= bboxes['boxes'][1]:
                continue
            # remove boxes smaller than X% of old area
            bboxes['area'] = (bboxes['boxes'][2]-bboxes['boxes'][0])*(bboxes['boxes'][3]-bboxes['boxes'][1])
            if bboxes['area'] / old_area < min_visibility:
                continue
            new_bboxes.append(bboxes)
        if len(new_bboxes) == 0:
            new_target = {'boxes': torch.Tensor([]), 'area': torch.Tensor([])}
        elif len(new_bboxes) == 1:
            new_target = new_bboxes[0]
            new_target['boxes'] = new_target['boxes'].reshape(1, -1)
        else:
            new_target = {
                key: torch.cat([val[key].reshape(1) for val in new_bboxes]) if key != 'boxes'
                else torch.cat([val[key].reshape(1, -1) for val in new_bboxes])
                for key in keys 
            }
        new_target['image_id'] = target['image_id']
        return new_target


class SafeRandomCrop(RandomCrop):
    """
    Random crop that preserves at least one bounding box.
    """

    def forward(
        self,
        images: ImagesType,
        targets: TargetsType,
        output_size: int | tuple[int, int] | None = None,
    ) -> tuple[ImagesType, TargetsType]:
        assert self.output_size is not None or output_size is not None, f'Given {self.output_size = } and {output_size = }.'

        # override output size if provided
        output_size = self.output_size if output_size is None else output_size
        if isinstance(output_size, int):
            output_size = (output_size, output_size)

        images, targets = list(images), list(targets)
        for i, (image, target) in enumerate(zip(images, targets)):
            # find cropping coordinates
            _, h, w = image.shape 
            if w == output_size[0] and h == output_size[1]:
                continue
            # choose a random bounding box
            idx = np.random.randint(0, len(target['boxes']))
            x1, y1, x2, y2 = target['boxes'][idx]
            # crop to preserve the bounding box 
            lower_bound_x = int(max(0, min(x2-output_size[0], w-output_size[0], x1)))
            upper_bound_x = int(max(0, min(x1, w-output_size[0])))
            if lower_bound_x == upper_bound_x:
                x = int(float(upper_bound_x))
            else:
                x = np.random.randint(lower_bound_x, upper_bound_x)
            lower_bound_y = int(max(0, min(y2-output_size[1], h-output_size[1], y1)))
            upper_bound_y = int(max(0, min(y1, h-output_size[1])))
            if lower_bound_y == upper_bound_y:
                y = int(float(upper_bound_y))
            else:
                y = np.random.randint(lower_bound_y, upper_bound_y)
            # crop the image
            images[i] = image[:, y:y+output_size[1], x:x+output_size[0]]
            _, nh, nw = images[i].shape 
            assert output_size[0] == nw and output_size[1] == nh, f'Expected: {output_size}; found: {(nw, nh)}. Original: {(w, h)}. Crop: {(x, y)}.'
            # adjust targets
            new_target = self._adjust_target(target, output_size, x, y)
            targets[i] = new_target
            assert len(new_target['boxes'].shape) == 2, f'Output: {output_size}. Image: {(w, h)}. Crop: {(x, y)}.\nTarget: {target}.\nNew: {new_target}.'

        return tuple(images), tuple(targets)


class IntelligentSquareResize(MosaicSubTransform):

    def __init__(
        self,
        output_size: int,
        max_aspect_ratio: float | None = None,
        crop_to_square: bool = False,
        centre_pad: bool = False,
        min_visibility: float | None = None,
    ) -> None:
        """
        Resize padded and/or cropped images to minimise the amount of padding 
        in the final mosaic.

        Parameters
        ----------
        output_size: int
            The width and height of each individual image in the mosaic 
            (it must be a square, hence only `int` is allowed).
        max_aspect_ratio: float, optional
            If a `float`, then images with aspect ratio (defined as the greater side
            divided by the lesser side, thus, always greater than unity) greater than 
            `max_aspect_ratio` will be cropped to square before any other transform
            is applied.
        crop_to_square: bool
            If `True`, all images will be cropped to square before any other transform 
            is applied. If `True`, `max_aspect_ratio` is ignored. Defaults to `False`.
        centre_pad: bool
            See `SquarePad`.
        min_visibility: float, optional
            See `RandomCrop`.
        """
        super().__init__()
        self.output_size = output_size
        self.max_aspect_ratio = max_aspect_ratio
        self.crop_to_square = crop_to_square
        self.centre_pad = centre_pad
        self.min_visibility = min_visibility

        if not self.crop_to_square and self.max_aspect_ratio is None:
            raise ValueError(f'Both crop_to_square and max_aspect_ratio can not be None.')

        self.pad_and_resize = v2.Compose([SquarePad(self.centre_pad), Resize(self.output_size)])
        self.random_crop = SafeRandomCrop(-1, self.min_visibility)

    def forward(self, images: ImagesType, targets: TargetsType) -> tuple[ImagesType, TargetsType]:
        images, targets = list(images), list(targets)
        for i, (image, target) in enumerate(zip(images, targets)):
            _, h, w = image.shape
            if self.crop_to_square:
                if h == w: 
                    continue
                output_size = (min(h, w), min(h, w))
                image, target = self.random_crop((image,), (target,), output_size)
                images[i], targets[i] = image[0], target[0]
                continue
            aspect_ratio = max(h, w) / min(h, w)
            if self.max_aspect_ratio and aspect_ratio > self.max_aspect_ratio:
                greater_size = int(self.max_aspect_ratio * min(h, w))
                output_size = (greater_size, h) if w > h else (w, greater_size)
                image, target = self.random_crop((image,), (target,), output_size)
                images[i], targets[i] = image[0], target[0]
        images, targets = self.pad_and_resize(images, targets)

        return tuple(images), tuple(targets)


class TypeConverter(torch.nn.Module):
    """
    Converts images to `torch.float32`, as well as, target bounding boxes 
    to `tv_tensors.BoundingBoxes` and target area to `torch.int32`.
    """

    def forward(self, images: ImagesType, targets: TargetsType) -> tuple[ImagesType, TargetsType]:
        images, targets = list(images), list(targets)
        for i in range(len(images)):
            images[i] = images[i].to(torch.float32)
            targets[i]['boxes'] = tv_tensors.BoundingBoxes(
                targets[i]['boxes'],
                format='XYXY',
                canvas_size=tuple(images[i].shape[1:]),
            )
            targets[i]['area'] = targets[i]['area'].to(torch.int32)
            if len(targets[i]['labels'].shape) == 0:
                targets[i]['labels'] = targets[i]['labels'].reshape(1)

        return tuple(images), tuple(targets)


class Mosaic(torch.nn.Module):

    def __init__(
        self,
        output_size: int = 1_000,
        min_possible_image_area: float = 0.25,
        centre_pad: bool = False,
        crop: bool = True,
        min_visibility: bool | float = 0.05,
        intelligent_resize: bool = True,
        max_aspect_ratio: float | None = None,
        crop_to_square: bool = False,
    ) -> None:
        """
        Mosaic data augmentation.

        Parameters
        ----------
        output_size: int
            The size of the final image. Defaults to `1000`.
        min_possible_image_area: float
            The minimal fraction of any of the four images to be visiable in the cropped mosaic.
            This is added in order to avoid excessive cropping of any of the images.
            It is used to calculate pre-cropping size of the grid. Deafults to `0.25`.

            For example, if `output_size = 1000` and `min_possible_image_area = 0.25`, then, first,
            a `1500x1500` grid will be constructed (each image will be resized to `750x750`) and 
            then it will be randomly cropped to `1000x1000`.
            
            If `crop = False`, then a `1000x1000` grid will be constructed straight away.
        centre_pad: bool
            If `True`, the four images will be padded with equal padding on each side as needed 
            to create a square. Otherwise, the padding will be added such that no image has 
            padding separating it from the centre of the mosaic. Defaults to `False`.
            See `SquarePad`.
        crop: bool 
            If `True`, a larger grid will be constructed first before randomly cropping
            to the required size. Defaults to `True`.
        min_visibility: bool | float
            Parameter passed to `RandomCrop`. If not `False` and greater than `0.0`, the bounding
            boxes that after cropping have less than `min_visibility` fraction of area 
            visiable in the cropped image will be removed. This is done to avoid edge cases where 
            almost all of the bounding box resides outside of the cropped image. Defaults to `0.05`.
            See `RandomCrop`.
        intelligent_resize: bool
            If `False`, all images will be padded to square and resized. This may result in large amounts 
            of padding visiable in the final mosaic if the dataset has images of various aspect ratios.
            If `True`, images with aspect ratio (defined as the greater side divided by the lesser side,
            thus, always greater than unity) greater than `max_aspect_ratio` will be cropped to square 
            before any other transforms are applied. Defaults to `True`.
        max_aspect_ratio: float, optional
            If a `float`, then images with aspect ratio (defined as the greater side divided by the lesser
            side, thus, always greater than unity) greater than `max_aspect_ratio` will be cropped to
            square before any other transform is applied.
        crop_to_square: bool
            If `True`, all images will be cropped to square before any other transform is applied. 
            If `True`, `max_aspect_ratio` is ignored. Defaults to `False`.
        """
        super().__init__()
        self.output_size = output_size
        self.min_possible_image_area = min_possible_image_area
        self.centre_pad = centre_pad 
        self.crop = crop
        self.min_visibility = min_visibility
        self.intelligent_resize = intelligent_resize
        self.max_aspect_ratio = max_aspect_ratio  
        self.crop_to_square = crop_to_square 
        # the total size of the grid of 4 images
        self.grid_size = self.output_size * (2 - np.sqrt(self.min_possible_image_area)) if self.crop else self.output_size
        # the target size of an image
        self.images_size = int(self.grid_size / 2)
        # preprocessing transforms
        if self.intelligent_resize:
            self.preprocess = v2.Compose([IntelligentSquareResize(
                output_size=self.images_size,
                max_aspect_ratio=self.max_aspect_ratio,
                crop_to_square=self.crop_to_square,
                min_visibility=self.min_visibility,
            )])
        else:
            self.preprocess = v2.Compose([SquarePad(self.centre_pad), Resize(self.images_size)])
        # postprocessing transforms
        if crop:
            self.postprocess = v2.Compose([SafeRandomCrop(self.output_size, self.min_visibility), TypeConverter()])
        else:
            self.postprocess = v2.Compose([TypeConverter()])
    
    def forward(self, images: ImagesType, targets: TargetsType) -> tuple[ImagesType, TargetsType]:
        images, targets = list(images), list(targets)
        # resize the images to fit their quarters
        # odd images will be ignored
        n = (len(images)//4) * 4
        images[:n], targets[:n] = self.preprocess(images[:n], targets[:n])
        # join the images into a grid
        new_images, new_targets = [], []
        for batch_idx in grouper(range(len(images)), n=4, incomplete='ignore'):
            # join images
            img_batch = [images[i] for i in batch_idx]
            top = torch.cat(img_batch[:2], dim=2)
            bottom = torch.cat(img_batch[2:], dim=2)
            grid_images = torch.cat([top, bottom], dim=1)
            new_images.append(grid_images)
            # correct targets to fit the grid
            tar_batch = [targets[i] for i in batch_idx]
            assert set([len(target['boxes'].shape) for target in tar_batch]) == {2}, tar_batch
            for i, pos in enumerate(['tl', 'tr', 'bl', 'br']):
                # if len(tar_batch[i]['boxes']) == 0:
                #     continue
                if pos == 'tl':
                    continue
                elif pos == 'tr':
                    tar_batch[i]['boxes'][:, [0, 2]] += self.images_size
                elif pos == 'bl':
                    tar_batch[i]['boxes'][:, [1, 3]] += self.images_size
                elif pos == 'br':
                    tar_batch[i]['boxes'][:, [0, 2]] += self.images_size
                    tar_batch[i]['boxes'][:, [1, 3]] += self.images_size
            grid_targets = {}
            for key in tar_batch[0].keys():
                grid_targets[key] = torch.cat([
                    target[key].reshape(1) if len(target[key].shape) == 0 else target[key]
                    for target in tar_batch
                ])
            new_targets.append(grid_targets)
        # crop the images to output size and convert to proper dtypes
        new_images, new_targets = self.postprocess(new_images, new_targets)
        new_images, new_targets = list(new_images), list(new_targets)

        if (rem := divmod(len(images), 4)[1]) != 0:
            new_images += images[-rem:]
            new_targets += targets[-rem:]

        return tuple(new_images), tuple(new_targets)


class Identity(torch.nn.Module):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

    def forward(self, images: ImagesType, targets: TargetsType) -> tuple[ImagesType, TargetsType]:
        return images, targets


class ExtraAugPolicy(enum.StrEnum):
    MOSAIC = 'mosaic'
    IMAGENET_AUTOAUGMENT = 'imagenet_autoaugment'


class ExtraAugmentation:

    def __init__(self, policy: ExtraAugPolicy) -> None:
        self.policy = policy
        if self.policy == ExtraAugPolicy.MOSAIC:
            mosaic = Mosaic(
                output_size=1000,
                min_possible_image_area=0.25,
                centre_pad=False,
                crop=True,
                min_visibility=0.02,
                intelligent_resize=True,
                max_aspect_ratio=1.2,
                crop_to_square=False,
            )
            identity = Identity()
            self.transform = v2.RandomChoice([identity, mosaic])
        elif self.policy == ExtraAugPolicy.IMAGENET_AUTOAUGMENT:
            raise NotImplementedError('ExtraAugPolicy.IMAGENET_AUTOAUGMENT not implemented.')

    def __call__(self, images: ImagesType, targets: TargetsType) -> tuple[ImagesType, TargetsType]:
        return self.transform(images, targets)


class MyAutoAugment(AutoAugment):

    def __init__(self, interpolation: F.InterpolationMode = F.InterpolationMode.NEAREST, fill: list[float] | None = None) -> None:
        super().__init__(interpolation=interpolation, fill=fill)
        self.policy = 'AdaptedImageNet'
        self.policies = [
            # (('Posterize', 0.4, 8), ('Rotate', 0.6, 9)),
            (('Solarize', 0.6, 5), ('AutoContrast', 0.6, None)),
            (('Equalize', 0.8, None), ('Equalize', 0.6, None)),
            (('Posterize', 0.6, 7), ('Posterize', 0.6, 6)),
            (('Equalize', 0.4, None), ('Solarize', 0.2, 4)),
            # (('Equalize', 0.4, None), ('Rotate', 0.8, 8)),
            (('Solarize', 0.6, 3), ('Equalize', 0.6, None)),
            (('Posterize', 0.8, 5), ('Equalize', 1.0, None)),
            # (('Rotate', 0.2, 3), ('Solarize', 0.6, 8)),
            (('Equalize', 0.6, None), ('Posterize', 0.4, 6)),
            # (('Rotate', 0.8, 8), ('Color', 0.4, 0)),
            # (('Rotate', 0.4, 9), ('Equalize', 0.6, None)),
            (('Equalize', 0.0, None), ('Equalize', 0.8, None)),
            (('Invert', 0.6, None), ('Equalize', 1.0, None)),
            (('Color', 0.6, 4), ('Contrast', 1.0, 8)),
            # (('Rotate', 0.8, 8), ('Color', 1.0, 2)),
            (('Color', 0.8, 8), ('Solarize', 0.8, 7)),
            (('Sharpness', 0.4, 7), ('Invert', 0.6, None)),
            # (('ShearX', 0.6, 5), ('Equalize', 1.0, None)),
            (('Color', 0.4, 0), ('Equalize', 0.6, None)),
            (('Equalize', 0.4, None), ('Solarize', 0.2, 4)),
            (('Solarize', 0.6, 5), ('AutoContrast', 0.6, None)),
            (('Invert', 0.6, None), ('Equalize', 1.0, None)),
            (('Color', 0.6, 4), ('Contrast', 1.0, 8)),
            (('Equalize', 0.8, None), ('Equalize', 0.6, None)),
        ]

    def forward(self, images: ImagesType, targets: TargetsType) -> tuple[ImagesType, TargetsType]:
        images = super().forward(images)
        return images, targets


def get_transforms(apply_color_jitter: bool = False) -> v2.Compose:
    transforms = []
    if apply_color_jitter:
        transforms += [
            MyAutoAugment()
        ]
    transforms += [
        v2.ToDtype(torch.float32, scale=True),
    ]
    return v2.Compose(transforms)

