"""Apply Zoo/fine-tuned detectors through FiftyOne and retain per-image metrics."""
import argparse
import sys
import json
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.catalog import MODELS, resolve_model
from src.data.dataset import DEFAULT_COCO, ROOT, load_coco, as_fiftyone
from src.models.adapters import load_pretrained, load_finetuned

# Explicit coarse mapping for optional COCO baselines, never a substitute for fine-tuning.
COCO_TO_ZOD = {'car': 'Vehicle', 'bus': 'Vehicle', 'truck': 'Vehicle', 'motorcycle': 'VulnerableVehicle',
               'bicycle': 'VulnerableVehicle', 'person': 'Pedestrian'}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def detection_payload(container, classes, size, pretrained=False):
    width, height = size
    boxes, scores, labels = [], [], []
    for detection in container.detections:
        label = COCO_TO_ZOD.get(detection.label) if pretrained else detection.label
        if label is None:
            continue
        if label not in classes:
            raise ValueError(f'Unexpected prediction label {label!r}; check checkpoint category mapping')
        x, y, w, h = detection.bounding_box
        x1, y1, x2, y2 = max(0., x), max(0., y), min(1., x+w), min(1., y+h)
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append([x1*width, y1*height, x2*width, y2*height])
        scores.append(float(detection.confidence) if detection.confidence is not None else 1.)
        labels.append(classes.index(label))
    return {'boxes': boxes, 'scores': scores, 'labels': labels}


def score_payload(prediction, target):
    import torch
    from src.models.metrics import get_metrics
    def tensors(payload):
        return {'boxes': torch.tensor(payload['boxes'], dtype=torch.float32).reshape(-1, 4),
                'scores': torch.tensor(payload['scores'], dtype=torch.float32),
                'labels': torch.tensor(payload['labels'], dtype=torch.int64)}
    scores = get_metrics([tensors(prediction)], [tensors(target)], metrics=['iou', 'lrp'])
    return {**scores, 'confidence': prediction['scores']}


def run_split(args, family, model, split, manifest):
    data = load_coco(args.coco_dir, split, args.image_root, args.max_samples)
    if manifest and manifest['classes'] != data['classes']:
        raise ValueError('Checkpoint classes/order differ from the COCO dataset')
    dataset = as_fiftyone(data)
    try:
        # A fresh temporary dataset per split/model prevents stale predictions.
        dataset.apply_model(model, label_field='predictions', confidence_thresh=args.confidence_threshold,
            batch_size=args.batch_size, num_workers=args.workers, skip_failures=False)
        if dataset.exists('predictions').count() != len(data['images']):
            raise RuntimeError('Incomplete predictions; refusing to save partial results')
        predictions, detections, discretized = {}, {}, {}
        for sample in dataset.iter_samples():
            image_id = str(sample.image_id)
            prediction = detection_payload(sample.predictions, data['classes'], (1344, 768), args.pretrained)
            target = detection_payload(sample.ground_truth, data['classes'], (1344, 768))
            scores = score_payload(prediction, target)
            predictions[image_id], detections[image_id] = prediction, scores
            if args.discretize_threshold is not None:
                discretized[image_id] = {k: (float(v > args.discretize_threshold) if k in ('iou', 'lrp') and v is not None else v)
                                        for k, v in scores.items()}
        base = args.output_dir or ROOT / 'results'
        # Preserve results/<family>/detections.json for the existing feature merger.
        output = (base / 'pretrained' / family / split if args.pretrained else
                  base / family if split == 'test' else base / family / split)
        if args.max_samples is not None:
            output = output / f'partial-{args.max_samples}'
        atomic_json(output / 'predictions.json', predictions)
        atomic_json(output / 'detections.json', detections)
        if discretized:
            atomic_json(output / 'detections_disc.json', discretized)
        atomic_json(output / 'predictions.meta.json', {'model': MODELS[family].zoo_name,
            'checkpoint': manifest, 'pretrained': args.pretrained, 'split': split,
            'classes': data['classes'], 'label_ids': 'zero-based contiguous class indices',
            'box_format': 'xyxy', 'coordinate_size': [1344, 768],
            'confidence_threshold': args.confidence_threshold, 'sample_count': len(predictions),
            'partial': args.max_samples is not None, 'coco_dir': str(args.coco_dir.resolve()),
            'coarse_coco_mapping': COCO_TO_ZOD if args.pretrained else None})
        print(f'{family} / {split}: {len(predictions)} images saved to {output}')
    finally:
        dataset.delete()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model', nargs='?', help='Family, full Zoo name, or all')
    p.add_argument('checkpoint', nargs='?', type=Path, help='Artifact directory containing manifest.json')
    p.add_argument('--models', nargs='+', help='Run several families or Zoo names')
    p.add_argument('--list-models', action='store_true')
    p.add_argument('--pretrained', action='store_true', help='Explicit COCO baseline; saved separately')
    p.add_argument('--coco-dir', type=Path, default=DEFAULT_COCO)
    p.add_argument('--image-root', type=Path)
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--artifacts-dir', type=Path, default=ROOT / 'models')
    p.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    p.add_argument('--train', action='store_true')
    p.add_argument('--validate', action='store_true')
    p.add_argument('--test', action='store_true')
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--confidence-threshold', type=float, default=0.5)
    p.add_argument('--discretize-threshold', type=float)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--max-samples', type=int)
    args = p.parse_args(argv)
    if not args.list_models and not (args.model or args.models):
        p.error('Specify a model, --models, or --list-models')
    if args.model and args.models:
        p.error('Use the positional model or --models, not both')
    if args.checkpoint and (args.pretrained or args.model == 'all' or args.models):
        p.error('A checkpoint directory requires a single fine-tuned model')
    if not 0 <= args.confidence_threshold <= 1:
        p.error('--confidence-threshold must be in [0, 1]')
    if args.discretize_threshold is not None and not 0 <= args.discretize_threshold <= 1:
        p.error('--discretize-threshold must be in [0, 1]')
    if args.batch_size < 1 or args.workers < 0 or (args.max_samples is not None and args.max_samples < 1):
        p.error('Batch size and sample limit must be positive; workers must be nonnegative')
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.list_models:
        for family, spec in MODELS.items():
            print(f'{family:14s} {spec.zoo_name}')
        return
    names = args.models or (list(MODELS) if args.model == 'all' else [args.model])
    splits = [s for s, enabled in [('train', args.train), ('val', args.validate), ('test', args.test)] if enabled] if any([args.train, args.validate, args.test]) else [args.split]
    for name in names:
        family, _ = resolve_model(name)
        if args.pretrained:
            model, manifest = load_pretrained(family, args.device), None
        else:
            artifact = args.checkpoint or args.artifacts_dir / family
            model, manifest = load_finetuned(artifact, args.device)
            if manifest['family'] != family:
                raise ValueError(f'{artifact} is for {manifest["family"]}, not {family}')
        for split in splits:
            run_split(args, family, model, split, manifest)
        del model
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
