import enum

import torch 
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection import fasterrcnn_resnet50_fpn 
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

import constants as const


class ModelExtendMode(enum.StrEnum):
    APPEND = 'append'
    REPLACE = 'replace'


class CosineSimilarityPredictor(torch.nn.Module):

    def __init__(self, in_channels: int, num_classes: int, scale: float = const.COSINE_SCALE) -> None:
        super().__init__()
        self.cls_score = torch.nn.Linear(in_channels, num_classes, bias=False)
        self.bbox_pred = torch.nn.Linear(in_channels, num_classes * 4)
        self.scale = torch.nn.Parameter(torch.ones(1)*scale)

        torch.nn.init.normal_(self.cls_score.weight, std=0.01)
        torch.nn.init.normal_(self.bbox_pred.weight, std=0.001)
        torch.nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 4:
            torch._assert(
                list(x.shape[2:]) == [1, 1],
                f'x has the wrong shape, expecting the last two dimensions to be [1,1] instead of {list(x.shape[2:])}',
            )
        x = x.flatten(start_dim=1)

        # normalize the input x along the `input_size` dimension
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 1e-5)

        # normalize weight
        temp_norm = (
            torch.norm(self.cls_score.weight.data, p=2, dim=1)
            .unsqueeze(1)
            .expand_as(self.cls_score.weight.data)
        )
        self.cls_score.weight.data = self.cls_score.weight.data.div(temp_norm+1e-5)
        cos_dist = self.cls_score(x_normalized)
        scores = self.scale * cos_dist

        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


class TwoLayeredFastRCNNPredictor(torch.nn.Module):

    def __init__(self, in_channels: int, num_classes_1: int, num_classes_2: int) -> None:
        super().__init__()
        self.cls_score_1 = torch.nn.Linear(in_channels, num_classes_1)
        self.bbox_pred_1 = torch.nn.Linear(in_channels, num_classes_1 * 4)
        self.cls_score_2 = torch.nn.Linear(num_classes_1, num_classes_2)
        self.bbox_pred_2 = torch.nn.Linear(num_classes_1 * 4, num_classes_2 * 4)

        torch.nn.init.normal_(self.cls_score_1.weight, std=0.01)
        torch.nn.init.normal_(self.bbox_pred_1.weight, std=0.001)
        torch.nn.init.constant_(self.cls_score_1.bias, 0)
        torch.nn.init.constant_(self.bbox_pred_1.bias, 0)
        torch.nn.init.normal_(self.cls_score_2.weight, std=0.01)
        torch.nn.init.normal_(self.bbox_pred_2.weight, std=0.001)
        torch.nn.init.constant_(self.cls_score_2.bias, 0)
        torch.nn.init.constant_(self.bbox_pred_2.bias, 0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 4:
            torch._assert(
                list(x.shape[2:]) == [1, 1],
                f'x has the wrong shape, expecting the last two dimensions to be [1,1] instead of {list(x.shape[2:])}',
            )
        x = x.flatten(start_dim=1)
        scores = self.cls_score_2(self.cls_score_1(x))
        bbox_deltas = self.bbox_pred_2(self.bbox_pred_1(x))

        return scores, bbox_deltas


class TwoLayeredCosineSimilarityPredictor(torch.nn.Module):

    def __init__(self, in_channels: int, num_classes_1: int, num_classes_2: int, scale: float = const.COSINE_SCALE) -> None:
        super().__init__()
        self.cls_score_1 = torch.nn.Linear(in_channels, num_classes_1)
        self.bbox_pred_1 = torch.nn.Linear(in_channels, num_classes_1 * 4)
        self.cls_score_2 = torch.nn.Linear(num_classes_1, num_classes_2, bias=False)
        self.bbox_pred_2 = torch.nn.Linear(num_classes_1 * 4, num_classes_2 * 4)

        torch.nn.init.normal_(self.cls_score_1.weight, std=0.01)
        torch.nn.init.normal_(self.bbox_pred_1.weight, std=0.001)
        torch.nn.init.constant_(self.cls_score_1.bias, 0)
        torch.nn.init.constant_(self.bbox_pred_1.bias, 0)
        torch.nn.init.normal_(self.cls_score_2.weight, std=0.01)
        torch.nn.init.normal_(self.bbox_pred_2.weight, std=0.001)
        torch.nn.init.constant_(self.bbox_pred_2.bias, 0)

        self.scale = torch.nn.Parameter(torch.ones(1)*scale)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 4:
            torch._assert(
                list(x.shape[2:]) == [1, 1],
                f'x has the wrong shape, expecting the last two dimensions to be [1,1] instead of {list(x.shape[2:])}',
            )
        x = x.flatten(start_dim=1)

        # bounding box regressor
        xb = self.bbox_pred_1(x)
        xb = xb.flatten(start_dim=1)
        bbox_deltas = self.bbox_pred_2(xb)

        # classifier
        xc = self.cls_score_1(x)
        xc = xc.flatten(start_dim=1)
        # normalize the input x along the `input_size` dimension
        xc_norm = torch.norm(xc, p=2, dim=1).unsqueeze(1).expand_as(xc)
        xc_normalized = xc.div(xc_norm + 1e-5)
        # normalize weight
        temp_norm = (
            torch.norm(self.cls_score_2.weight.data, p=2, dim=1)
            .unsqueeze(1)
            .expand_as(self.cls_score_2.weight.data)
        )
        self.cls_score_2.weight.data = self.cls_score_2.weight.data.div(temp_norm+1e-5)
        cos_dist = self.cls_score_2(xc_normalized)
        scores = self.scale * cos_dist

        return scores, bbox_deltas


def get_default_model(num_classes: int | None = None) -> FasterRCNN:
    """
    Returns Faster-RCNN model with FPN and ResNet-50 backbone. If `num_classes`
    is passed, the box classifier and regressor will be replaced with 
    randomly initialized weights.
    """
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    if num_classes is None:
        return model
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    predictor = FastRCNNPredictor(in_features, num_classes)
    model.roi_heads.box_predictor = predictor
    return model
    


def extend_model_to_fsod(
    model: FasterRCNN, 
    novel_weights_fp: str | None = None, 
    use_cosine: bool = False,
    cppe_only: bool = False,
    mode: ModelExtendMode = ModelExtendMode.REPLACE,
) -> FasterRCNN:
    """
    Parameters
    ----------
    model: FasterRCNN
        The model to modify.
    novel_weights_fp: str, optional
        The weights corresponding to the base classes are set as those obtained in the previous stage, 
        and the weights corresponding to the novel classes are randomly initialized if 
        `novel_weights_fp` is `None` or those of a predictor fine-tuned on the novel set otherwise.
    use_cosine: bool
        If `True`, cosing similarity for box classifiers will be used.
        Default `False`.
    cppe_only: bool
        If `True`, the output layer will be for CPPE-5 categories only.
        Default `False`.
    """
    if mode == ModelExtendMode.REPLACE:
        model = replace_last_layer(model, novel_weights_fp, use_cosine, cppe_only) 
    else:
        model = append_last_layer(model, novel_weights_fp, use_cosine, cppe_only) 
    return model


def replace_last_layer(
    model: FasterRCNN, 
    novel_weights_fp: str | None = None, 
    use_cosine: bool = False,
    cppe_only: bool = False,
) -> FasterRCNN:
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    if cppe_only:
        out_features = const.CPPE_NUM_CATEGORIES + 1
    else:
        out_features = const.COCO_NUM_CATEGORIES + const.CPPE_NUM_CATEGORIES + 1

    if novel_weights_fp is None:
        if cppe_only:
            num_classes = const.CPPE_NUM_CATEGORIES+1
        else:
            num_classes = const.CPPE_NUM_CATEGORIES
        cls_score_extension = torch.nn.init.xavier_uniform_(torch.empty((num_classes, in_features)))
        bbox_pred_extension = torch.nn.init.xavier_uniform_(torch.empty((num_classes*4, in_features)))
    else:
        novel_weights = torch.load(novel_weights_fp, map_location=torch.device('cpu'))
        if cppe_only:
            cls_score_extension = novel_weights['roi_heads.box_predictor.cls_score.weight'].data
            bbox_pred_extension = novel_weights['roi_heads.box_predictor.bbox_pred.weight'].data
        else:
            cls_score_extension = novel_weights['roi_heads.box_predictor.cls_score.weight'].data[1:]
            bbox_pred_extension = novel_weights['roi_heads.box_predictor.bbox_pred.weight'].data[4:]

    if cppe_only:
        new_weights_cls_score = cls_score_extension
        new_weights_bbox_pred = bbox_pred_extension
    else:
        old_weights_cls_score = model.roi_heads.box_predictor.cls_score.weight.data
        new_weights_cls_score = torch.concat((old_weights_cls_score, cls_score_extension))
        old_weights_bbox_pred = model.roi_heads.box_predictor.bbox_pred.weight.data
        new_weights_bbox_pred = torch.concat((old_weights_bbox_pred, bbox_pred_extension))

    if use_cosine:
        predictor = CosineSimilarityPredictor(in_features, out_features)
    else:
        predictor = FastRCNNPredictor(in_features, out_features)
        torch.nn.init.normal_(predictor.cls_score.weight, std=0.01)
        torch.nn.init.normal_(predictor.bbox_pred.weight, std=0.001)
        torch.nn.init.constant_(predictor.bbox_pred.bias, 0)
    model.roi_heads.box_predictor = predictor

    model.roi_heads.box_predictor.cls_score.weight.data = new_weights_cls_score
    model.roi_heads.box_predictor.bbox_pred.weight.data = new_weights_bbox_pred

    return model


def append_last_layer(
    model: FasterRCNN, 
    novel_weights_fp: str | None = None, 
    use_cosine: bool = False,
    cppe_only: bool = False,
) -> FasterRCNN:
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    intermediate_feature = const.COCO_NUM_CATEGORIES + 1
    if cppe_only:
        out_features = const.CPPE_NUM_CATEGORIES + 1
    else:
        out_features = const.COCO_NUM_CATEGORIES + const.CPPE_NUM_CATEGORIES + 1

    old_weights_cls_score = model.roi_heads.box_predictor.cls_score.weight.data
    old_weights_bbox_pred = model.roi_heads.box_predictor.bbox_pred.weight.data

    if novel_weights_fp is None:
        cls_score = torch.nn.init.xavier_uniform_(torch.empty((out_features, old_weights_cls_score.shape[0])))
        bbox_pred = torch.nn.init.xavier_uniform_(torch.empty((out_features*4, old_weights_bbox_pred.shape[0])))
    else:
        novel_weights = torch.load(novel_weights_fp, map_location=torch.device('cpu'))
        cls_score = novel_weights['roi_heads.box_predictor.cls_score.weight'].data[1:]
        bbox_pred = novel_weights['roi_heads.box_predictor.bbox_pred.weight'].data[4:]

    if use_cosine:
        predictor = TwoLayeredCosineSimilarityPredictor(in_features, intermediate_feature, out_features)
    else:
        predictor = TwoLayeredFastRCNNPredictor(in_features, intermediate_feature, out_features)
    model.roi_heads.box_predictor = predictor

    model.roi_heads.box_predictor.cls_score_1.weight.data = old_weights_cls_score
    model.roi_heads.box_predictor.bbox_pred_1.weight.data = old_weights_bbox_pred
    model.roi_heads.box_predictor.cls_score_2.weight.data = cls_score
    model.roi_heads.box_predictor.bbox_pred_2.weight.data = bbox_pred

    return model


def freeze_for_fine_tuning(
    model: FasterRCNN,
    mode: ModelExtendMode = ModelExtendMode.REPLACE,
    freeze_all: bool = False,
) -> FasterRCNN:
    if freeze_all:
        for param in model.parameters():
            param.requires_grad = False
    else:
        if mode == ModelExtendMode.REPLACE:
            params_to_train = [
                'roi_heads.box_predictor.cls_score.weight',
                'roi_heads.box_predictor.cls_score.bias',
                'roi_heads.box_predictor.bbox_pred.weight',
                'roi_heads.box_predictor.bbox_pred.bias',
            ]
        else:
            params_to_train = [
                'roi_heads.box_predictor.cls_score_2.weight',
                'roi_heads.box_predictor.cls_score_2.bias',
                'roi_heads.box_predictor.bbox_pred_2.weight',
                'roi_heads.box_predictor.bbox_pred_2.bias',
            ]
        for name, param in model.named_parameters():
            param.requires_grad = True if name in params_to_train else False
    return model

