"""FiftyOne loaders for pretrained Zoo models and local fine-tuned artifacts."""
import json
from pathlib import Path

from src.models.catalog import MODELS


def zoo_loader_name(spec):
    import fiftyone.zoo as foz
    available = foz.list_zoo_models()
    if spec.zoo_name in available:
        return spec.zoo_name
    # FiftyOne 1.16 predates the named RT-DETR-v2 entry. Its generic
    # Transformers Zoo wrapper loads the same architecture from HF config.
    if spec.backend == 'transformers' and 'detection-transformer-torch' in available:
        return 'detection-transformer-torch'
    raise RuntimeError(f'{spec.zoo_name} is unavailable in the installed FiftyOne catalog')


def load_pretrained(family, device):
    import fiftyone.zoo as foz
    spec = MODELS[family]
    if spec.backend == 'rfdetr' and device.startswith('cuda:'):
        import torch
        torch.cuda.set_device(int(device.split(':')[1]))
        device = 'cuda'
    options = {'device': device}
    if spec.backend == 'transformers':
        options['name_or_path'] = spec.source
    return foz.load_zoo_model(zoo_loader_name(spec), **options)


def replace_torchvision_head(model, family, num_classes):
    if family == 'faster-rcnn':
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, num_classes + 1)
    else:
        from torchvision.models.detection.retinanet import RetinaNetClassificationHead
        tower = model.head.classification_head.conv.state_dict()
        classifier = RetinaNetClassificationHead(model.backbone.out_channels,
            model.anchor_generator.num_anchors_per_location()[0], num_classes + 1)
        classifier.conv.load_state_dict(tower)
        model.head.classification_head = classifier
        model.score_thresh = .001
    return model


def build_torchvision(family, num_classes):
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn
    constructor = fasterrcnn_resnet50_fpn if family == 'faster-rcnn' else retinanet_resnet50_fpn
    model = constructor(weights=None, weights_backbone=None)
    return replace_torchvision_head(model, family, num_classes)


def load_finetuned(artifact, device):
    artifact = Path(artifact).resolve()
    manifest = json.loads((artifact / 'manifest.json').read_text())
    family, classes = manifest['family'], manifest['classes']
    if family not in MODELS or manifest['zoo_name'] != MODELS[family].zoo_name:
        raise ValueError('Checkpoint architecture does not match the model catalog')
    checkpoint = artifact / manifest['checkpoint']
    backend = MODELS[family].backend
    if backend == 'ultralytics':
        from ultralytics import YOLO
        from fiftyone.utils.ultralytics import (FiftyOneYOLOModel, FiftyOneYOLOModelConfig,
                                               UltralyticsDetectionOutputProcessor)
        native = YOLO(str(checkpoint))
        if list(native.names.values()) != classes:
            raise ValueError('YOLO checkpoint classes differ from manifest')
        native.to(device)
        model = FiftyOneYOLOModel(FiftyOneYOLOModelConfig({'model': native, 'model_path': str(checkpoint), 'device': device,
            'output_processor_cls': UltralyticsDetectionOutputProcessor}))
    elif backend == 'transformers':
        import fiftyone.zoo as foz
        model = foz.load_zoo_model(zoo_loader_name(MODELS[family]), name_or_path=str(checkpoint), device=device)
    elif backend == 'rfdetr':
        import rfdetr
        from fiftyone.utils.rfdetr import RFDETRDetectionModel, RFDETRDetectionModelConfig
        class NativeRFDETR(getattr(rfdetr, MODELS[family].source)):
            @property
            def class_names(self):
                return dict(enumerate(classes))
        class LocalRFDETRModel(RFDETRDetectionModel):
            def _load_model(self, config):
                return NativeRFDETR(pretrain_weights=str(checkpoint), device=device,
                    group_detr=manifest['settings']['recipe'].get('group_detr', 13))
        model = LocalRFDETRModel(RFDETRDetectionModelConfig({'model_type': MODELS[family].source, 'device': device}))
    else:
        import torch
        from fiftyone.utils.torch import TorchImageModel, TorchImageModelConfig
        native = build_torchvision(family, len(classes))
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        native.load_state_dict(state['model'])
        model = TorchImageModel(TorchImageModelConfig({'model': native,
            'classes': ['background'] + classes, 'filter_classes': classes, 'device': device,
            'output_processor_cls': 'fiftyone.utils.torch.DetectorOutputProcessor',
            'raw_inputs': False}))
    return model, manifest


def save_manifest(output_dir, family, classes, checkpoint, settings):
    output = Path(output_dir).resolve()
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f'Training produced no checkpoint at {checkpoint}')
    manifest = {'family': family, 'zoo_name': MODELS[family].zoo_name, 'classes': classes,
                'checkpoint': str(checkpoint.relative_to(output)), 'settings': settings,
                'source': MODELS[family].source, 'zoo_loader': zoo_loader_name(MODELS[family])}
    from importlib.metadata import version
    manifest['versions'] = {p: version(p) for p in ['fiftyone', 'torch', 'torchvision']}
    temporary = output / 'manifest.json.tmp'
    temporary.write_text(json.dumps(manifest, indent=2))
    temporary.replace(output / 'manifest.json')
