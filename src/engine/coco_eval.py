import io
import copy
from contextlib import redirect_stdout

import torch
import numpy as np
from _pycocotools.coco import COCO
from _pycocotools.cocoeval import COCOeval


class CocoEvaluator:

    def __init__(self, coco_gt: COCO) -> None:
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt
        self.coco_eval = COCOeval(coco_gt, iouType='bbox') 
        self.img_ids = []
        self.eval_imgs = []

    def update(self, predictions: dict[int, list[dict]]) -> None:
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        results = self.prepare(predictions)

        # annsImgIds = [ann['image_id'] for ann in results]
        # assert set(annsImgIds) == (set(annsImgIds) & set(self.coco_gt.getImgIds())), 'Results do not correspond to current coco set'

        with redirect_stdout(io.StringIO()):
            coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()
        coco_eval = self.coco_eval

        coco_eval.cocoDt = coco_dt
        coco_eval.params.imgIds = list(img_ids)
        img_ids, eval_imgs = evaluate(coco_eval)

        self.eval_imgs.append(eval_imgs)

    def synchronize_between_processes(self) -> None:
        self.eval_imgs = np.concatenate(self.eval_imgs, 2)
        self.img_ids = list(self.img_ids)
        self.eval_imgs = list(self.eval_imgs.flatten())
        self.coco_eval.evalImgs = self.eval_imgs
        self.coco_eval.params.imgIds = self.img_ids
        self.coco_eval._paramsEval = copy.deepcopy(self.coco_eval.params)

    def accumulate(self) -> None:
        self.coco_eval.accumulate()

    def summarize(self) -> None:
        print(f'IoU metric:')
        self.coco_eval.summarize()

    def prepare(self, predictions: dict[int, list[dict]]) -> list[dict]:
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction['boxes']
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction['scores'].tolist()
            labels = prediction['labels'].tolist()

            coco_results.extend(
                [
                    {
                        'image_id': original_id,
                        'category_id': labels[i],
                        'bbox': box,
                        'score': scores[i],
                    }
                    for i, box in enumerate(boxes)
                ]
            )
        return coco_results


def convert_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def evaluate(imgs: COCOeval) -> tuple[list[int], np.ndarray]:
    with redirect_stdout(io.StringIO()):
        imgs.evaluate()
    return imgs.params.imgIds, np.asarray(imgs.evalImgs).reshape(-1, len(imgs.params.areaRng), len(imgs.params.imgIds))

