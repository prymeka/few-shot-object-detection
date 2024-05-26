import torch 
from torchvision import tv_tensors
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torchvision.datasets.coco import CocoDetection

import constants as const
from engine.utils import collate_fn


class CocoDataset(CocoDetection):

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict]:
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        width = image.width
        height = image.height
        num_objs = len(target)
        boxes, labels, img_id, is_crowd, areas = [], [], [], [], []
        for i in range(num_objs):
            labels.append(target[i]['category_id'])
            img_id.append(target[i]['image_id'])
            is_crowd.append(target[i]['iscrowd'])
            areas.append(target[i]['area'])
            # bounding boxes for objects:
            # in coco format, bbox = [xmin, ymin, width, height]
            # in pytorch, the input should be [xmin, ymin, xmax, ymax]
            x, y, w, h = target[i]['bbox']
            xmin = x
            ymin = y
            xmax = xmin + w
            ymax = ymin + h
            assert xmin >= 0
            assert xmax <= width
            assert xmin < xmax
            assert ymin >= 0
            assert ymax <= height
            assert ymin < ymax
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = tv_tensors.BoundingBoxes(boxes, format='XYXY', canvas_size=image.size)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        img_id = torch.as_tensor(list(set(img_id)), dtype=torch.int64)
        is_crowd = torch.as_tensor(is_crowd, dtype=torch.int8)
        areas = torch.as_tensor(areas, dtype=torch.float64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': img_id,
            'iscrowd': is_crowd,
            'area': areas
        }

        if self.transforms is not None:
            image = tv_tensors.Image(image)
            image, target = self.transforms(image, target)

        return image, target


def get_datasets(
    filepath: str,
    root: str = const.DATA_DIR,
    batch_size: int = const.BATCH_SIZE,
    transforms: v2.Compose | None = None,
    loader: bool = True,
) -> CocoDataset | DataLoader:
    result = CocoDataset(
        root=root,
        annFile=filepath,
        transforms=transforms,
    )
    if loader:
        result = DataLoader(
            result,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=True,
            num_workers=16,
        )
    return result

