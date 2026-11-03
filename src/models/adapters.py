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
        model.roi_heads.score_thresh = .001
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
                    num_classes=len(classes), resolution=manifest['settings']['recipe'].get('resolution', 560),
                    group_detr=manifest['settings']['recipe'].get('group_detr', 13))
        model = LocalRFDETRModel(RFDETRDetectionModelConfig({'model_type': MODELS[family].source, 'device': device}))
    else:
        import torch
        from fiftyone.utils.torch import TorchImageModel, TorchImageModelConfig
        native = build_torchvision(family, len(classes))
        recipe = manifest['settings']['recipe']
        native.transform.min_size = tuple(recipe.get('min_size', native.transform.min_size))
        native.transform.max_size = recipe.get('max_size', native.transform.max_size)
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


def set_resolution(model, family, resolution):
    backend = MODELS[family].backend
    if backend == 'ultralytics':
        predictor = model._model.predictor
        predictor.args.imgsz = resolution
        predictor.imgsz = [resolution, resolution]
        predictor.args.rect = True
    elif backend == 'rfdetr':
        model._model.model.resolution = resolution
    elif backend == 'transformers':
        processor = model.transforms.processor
        processor.size = ({'shortest_edge': resolution, 'longest_edge': round(resolution*864/512)}
                          if 'shortest_edge' in processor.size else {'height': resolution, 'width': resolution})
    else:
        model._model.transform.min_size = (resolution,)
        model._model.transform.max_size = round(resolution*1333/800)


def batch_predictor(family, artifact, resolution):
    import torch
    from torchvision.transforms.functional import pil_to_tensor

    manifest = json.loads((artifact/'manifest.json').read_text())
    checkpoint = artifact/manifest['checkpoint']
    if family == 'yolo':
        from ultralytics import YOLO
        model = YOLO(checkpoint)
        def predict(images):
            results = model.predict(images, imgsz=resolution, conf=.001, iou=.7, max_det=300,
                                   half=False, device='cuda', verbose=False)
            return [{'boxes': r.boxes.xyxy, 'scores': r.boxes.conf, 'labels': r.boxes.cls.long()} for r in results]
    elif family == 'rf-detr':
        import rfdetr
        model = rfdetr.RFDETRLarge(pretrain_weights=str(checkpoint), device='cuda', num_classes=3,
                                  group_detr=manifest['settings']['recipe'].get('group_detr', 13))
        model.model.resolution = resolution
        def predict(images):
            with torch.autocast('cuda', enabled=False):
                results = model.predict(images, threshold=.001)
            predictions = []
            for result in results:
                keep = result.class_id < 3
                predictions.append({'boxes': torch.as_tensor(result.xyxy[keep]),
                    'scores': torch.as_tensor(result.confidence[keep]),
                    'labels': torch.as_tensor(result.class_id[keep]).long()})
            return predictions
    elif MODELS[family].backend == 'transformers':
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
        model = AutoModelForObjectDetection.from_pretrained(checkpoint).cuda().eval()
        processor = AutoImageProcessor.from_pretrained(checkpoint)
        processor.size = ({'shortest_edge': resolution, 'longest_edge': round(resolution*864/512)}
                          if 'shortest_edge' in processor.size else {'height': resolution, 'width': resolution})
        def predict(images):
            inputs = processor(images=images, return_tensors='pt').to('cuda')
            with torch.autocast('cuda', enabled=False):
                result = model(**inputs)
            return processor.post_process_object_detection(result, threshold=.001,
                target_sizes=torch.tensor([[im.height, im.width] for im in images], device='cuda'))
    else:
        model = build_torchvision(family, 3)
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['model'])
        model = model.cuda().eval()
        if family == 'faster-rcnn':
            model.roi_heads.score_thresh = .001
        def predict(images):
            with torch.autocast('cuda', enabled=False):
                results = model([pil_to_tensor(im).cuda().float()/255 for im in images])
            predictions = []
            for result in results:
                keep = result['labels'] > 0
                predictions.append({k: v[keep] - 1 if k == 'labels' else v[keep] for k,v in result.items()})
            return predictions
    return predict

