"""One current closed-set Zoo representative per detector family."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    zoo_name: str
    backend: str
    source: str = ""


MODELS = {
    "yolo": ModelSpec("yolo26l-coco-torch", "ultralytics", "yolo26l.pt"),
    "rf-detr": ModelSpec("rfdetr-large-coco-torch", "rfdetr", "RFDETRLarge"),
    "rt-detr": ModelSpec("rtdetr-v2-l-coco-torch", "transformers", "PekingU/rtdetr_v2_r101vd"),
    "d-fine": ModelSpec("dfine-xlarge-coco-torch", "transformers", "ustc-community/dfine-xlarge-coco"),
    "detr": ModelSpec("detection-transformer-torch", "transformers", "facebook/detr-resnet-50"),
    "faster-rcnn": ModelSpec("faster-rcnn-resnet50-fpn-coco-torch", "torchvision"),
    "retinanet": ModelSpec("retinanet-resnet50-fpn-coco-torch", "torchvision"),
}


def resolve_model(name):
    if name in MODELS:
        return name, MODELS[name]
    for family, spec in MODELS.items():
        if name == spec.zoo_name:
            return family, spec
    raise ValueError(f"Unknown model {name!r}; choose from {', '.join(MODELS)}")
