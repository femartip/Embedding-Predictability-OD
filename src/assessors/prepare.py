"""Create a grouped image split and rescore frozen detector predictions."""
import argparse
import hashlib
import json
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torchvision.ops import box_iou, box_area
from tqdm import tqdm

from src.data.dataset import DEFAULT_COCO, load_coco, clipped_annotations, validate_splits
from src.models.metrics import LocalizationRecallPrecision
from .data import DEFAULT_DATA, write_json


def payload(path):
    if Path(path).suffix == '.jsonl':
        with Path(path).open() as stream:
            return {int(row['image_id']): row for row in map(json.loads, stream)}
    data = json.loads(Path(path).read_text())
    return {int(k): v for k, v in data.items()}


def prediction_rows(path):
    if Path(path).suffix == '.jsonl':
        with Path(path).open() as stream:
            for row in map(json.loads, stream):
                yield int(row['image_id']), row
    else:
        yield from payload(path).items()


def score(pred, image, annotations, categories, size, confidence):
    sx, sy = (size[0] / image['width'], size[1] / image['height']) if size else (1., 1.)
    anns = clipped_annotations(image, annotations)
    boxes = torch.tensor([[a['bbox'][0]*sx, a['bbox'][1]*sy,
                           (a['bbox'][0]+a['bbox'][2])*sx,
                           (a['bbox'][1]+a['bbox'][3])*sy] for a in anns]).reshape(-1, 4)
    labels = torch.tensor([categories[a['category_id']] for a in anns], dtype=torch.long)
    crowd = torch.tensor([bool(a.get('iscrowd', 0)) for a in anns], dtype=torch.bool)
    p = {'boxes': torch.tensor(pred['boxes'], dtype=torch.float32).reshape(-1, 4),
         'labels': torch.tensor(pred['labels'], dtype=torch.long),
         'scores': torch.tensor(pred['scores'], dtype=torch.float32)}
    keep = p['scores'] >= confidence
    p = {k: v[keep] for k, v in p.items()}
    # Ignore predictions on crowd boxes only when no valid same-class match exists.
    if crowd.any() and len(p['boxes']):
        a, b = p['boxes'], boxes[crowd]
        intersection = (torch.minimum(a[:, None, 2:], b[:, 2:]) -
                        torch.maximum(a[:, None, :2], b[:, :2])).clamp(min=0).prod(-1)
        on_crowd = ((intersection / box_area(a)[:, None].clamp(min=1e-6) >= .5) &
                    (p['labels'][:, None] == labels[crowd])).any(1)
        valid = box_iou(a, boxes[~crowd]) * (p['labels'][:, None] == labels[~crowd])
        has_match = (valid >= .5).any(1)
        p = {k: v[~on_crowd | has_match] for k, v in p.items()}
    target = {'boxes': boxes[~crowd], 'labels': labels[~crowd]}
    loc, fp, fn, tp = LocalizationRecallPrecision()._process_single(p, target)
    total = tp + fp + fn
    validity = 1 - float((loc / .5 + fp + fn) / total) if total else 1.
    return {'lrp': validity, 'fp_rate': float(fp / (tp+fp)) if tp+fp else 0.,
                     'fn_rate': float(fn / (tp+fn)) if tp+fn else 0.,
                     'localization_error': float(loc / tp) if tp else 1.,
                     'fp': float(fp), 'fn': float(fn), 'tp': float(tp)}


def predict_pool(family, artifact, resolution, data, output, batch_size):
    from PIL import Image
    from src.models.adapters import batch_predictor
    path = output / family / 'predictions.jsonl'
    metadata = path.with_name('predictions.meta.json')
    if metadata.exists():
        saved = json.loads(metadata.read_text())
        assert saved['artifact'] == artifact and saved['resolution'] == resolution
        assert saved['images'] == len(data['images'])
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    predict = batch_predictor(family, Path(artifact), resolution)
    with path.open('w') as stream, torch.inference_mode():
        for start in tqdm(range(0, len(data['images']), batch_size), desc=f'{family} predictions'):
            rows = data['images'][start:start+batch_size]
            images = []
            for row in rows:
                with Image.open(row['filepath']) as image:
                    assert image.size == (row['width'], row['height'])
                    images.append(image.convert('RGB'))
            predictions = predict(images)
            assert len(predictions) == len(rows)
            for row, pred in zip(rows, predictions):
                pred = {k: v.detach().cpu() for k, v in pred.items()}
                assert all(torch.isfinite(v).all() for v in pred.values())
                assert set(pred['labels'].tolist()) <= {0, 1, 2}
                pred['boxes'][:, 2:] = torch.maximum(pred['boxes'][:, 2:], pred['boxes'][:, :2])
                stream.write(json.dumps({'image_id': row['id'], **{k: v.tolist() for k, v in pred.items()}})+'\n')
            write_json(path.with_name('progress.json'), {'processed': min(start+batch_size, len(data['images'])),
                       'total': len(data['images'])})
    write_json(metadata, {'split': 'test', 'pretrained': False, 'partial': False,
        'coordinate_size': None, 'coordinate_space': 'original source pixels', 'classes': data['classes'],
        'confidence_threshold': .001, 'artifact': artifact, 'resolution': resolution, 'precision': 'FP32',
        'images': len(data['images'])})
    del predict
    gc.collect()
    torch.cuda.empty_cache()
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--groups', type=Path, required=True, help='CSV indexed by image ID; group column or lat/long')
    p.add_argument('--group-column', help='Use an existing drive/region column instead of geographic tiles')
    p.add_argument('--tile-size', type=float, default=.01)
    p.add_argument('--predictions', nargs='*', default=[], help='family=path/to/predictions.json')
    p.add_argument('--calibration', nargs='*', default=[], help='family=validation/predictions.json')
    p.add_argument('--thresholds', nargs='+', type=float, default=[.05, .1, .25, .5, .75])
    p.add_argument('--confidence', type=float, default=.5)
    p.add_argument('--coordinate-size', nargs=2, type=int, default=[1344, 768])
    p.add_argument('--coco-dir', type=Path, default=DEFAULT_COCO)
    p.add_argument('--image-root', type=Path)
    p.add_argument('--output', type=Path, default=DEFAULT_DATA)
    p.add_argument('--seed', type=int, default=43)
    p.add_argument('--selection', type=Path, help='Frozen detector selection.csv; reuse its full validation predictions')
    p.add_argument('--detectors', nargs='+', default=['yolo', 'rf-detr'])
    p.add_argument('--batch-size', type=int, default=8)
    args = p.parse_args()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    validate_splits(args.coco_dir)
    data = load_coco(args.coco_dir, 'test', args.image_root)
    rows = pd.DataFrame(data['images']).rename(columns={'id': 'image_id'})
    metadata = pd.read_csv(args.groups, index_col=0).loc[rows.image_id]
    assert metadata.index.is_unique and not metadata.isna().all(axis=1).any()
    if args.group_column:
        assert metadata[args.group_column].notna().all()
        rows['group'] = metadata[args.group_column].astype(str).to_numpy()
    else:
        coords = metadata[['lat', 'long']].to_numpy(float)
        assert np.isfinite(coords).all()
        rows['group'] = ['%d:%d' % tuple(c) for c in np.floor(coords / args.tile_size).astype(int)]
    rows['objects'] = [sum(not a.get('iscrowd', 0) for a in data['annotations'][i]) for i in rows.image_id]
    rows['mean_box_height'] = [np.mean([a['bbox'][3] for a in data['annotations'][i]]) for i in rows.image_id]
    for category in data['categories']:
        rows['objects_'+category['name']] = [sum(a['category_id'] == category['id'] and not a.get('iscrowd', 0)
                                                for a in data['annotations'][i]) for i in rows.image_id]
    manifest = args.output / 'images.csv'
    if manifest.exists():
        saved = pd.read_csv(manifest)
        assert saved.image_id.tolist() == rows.image_id.tolist()
        assert saved.group.astype(str).tolist() == rows.group.astype(str).tolist()
        rows = saved
    else:
        trainval, test = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=args.seed).split(rows, groups=rows.group))
        train, val = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=args.seed).split(rows.iloc[trainval], groups=rows.iloc[trainval].group))
        rows['split'] = 'test'
        rows.loc[trainval[train], 'split'] = 'train'
        rows.loc[trainval[val], 'split'] = 'val'
        rows.to_csv(manifest, index=False)
        write_json(args.output / 'split.json', {'seed': args.seed, 'groups': str(args.groups.resolve()),
            'group_column': args.group_column, 'tile_size_degrees': args.tile_size,
            'source': 'Repurposed detector test split; prior use must be audited',
            'counts': rows.split.value_counts().to_dict()})
    calibration = dict(s.split('=', 1) for s in args.calibration)
    if args.selection:
        selection = pd.read_csv(args.selection).set_index('family')
        args.coordinate_size = None
        args.predictions = []
        for family in args.detectors:
            selected = selection.loc[family]
            path = predict_pool(family, selected.artifact, int(selected.resolution), data,
                                args.output / 'predictions', args.batch_size)
            args.predictions.append(f'{family}={path}')
            calibration[family] = str(args.selection.parent / f'{family}-predictions.jsonl')
    targets = []
    provenance = {}
    for entry in args.predictions:
        family, path = entry.split('=', 1)
        confidence, curve = args.confidence, {}
        if family in calibration:
            val = load_coco(args.coco_dir, 'val', args.image_root)
            val_images = {im['id']: im for im in val['images']}
            seen = set()
            totals = np.zeros(len(args.thresholds))
            for image_id, pred in tqdm(prediction_rows(calibration[family]), total=len(val_images), desc=f'{family} calibration'):
                assert image_id not in seen
                seen.add(image_id)
                for i, threshold in enumerate(args.thresholds):
                    totals[i] += score(pred, val_images[image_id], val['annotations'][image_id],
                        val['category_to_index'], args.coordinate_size, threshold)['lrp']
            assert seen == val_images.keys()
            curve = {str(t): float(v/len(seen)) for t, v in zip(args.thresholds, totals)}
            confidence = float(max(curve, key=curve.get))
            print(f'{family}: confidence={confidence}, validation mean 1-LRP={curve}', flush=True)
        for prediction_path, expected_split in [(path, 'test')] + ([(calibration[family], 'val')] if family in calibration else []):
            meta_path = Path(prediction_path).with_name('predictions.meta.json')
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                assert meta['split'] == expected_split and not meta['pretrained'] and not meta['partial']
                assert meta['coordinate_size'] == args.coordinate_size and meta['classes'] == data['classes']
                minimum = min(args.thresholds) if expected_split == 'val' else confidence
                assert meta['confidence_threshold'] <= minimum, 'Regenerate predictions at a lower confidence floor'
            else:
                if args.selection and expected_split == 'val':
                    audit = json.loads((args.selection.parent / f'{family}.json').read_text())
                    assert audit['split'] == 'val' and audit['dtype'] == 'FP32'
                    assert audit['artifact'] == selection.loc[family].artifact
                    assert audit['resolution'] == int(selection.loc[family].resolution)
                    assert audit['coordinate_space'] == 'original source pixels'
                else:
                    print(f'{prediction_path}: legacy predictions; assuming declared coordinates, class order and confidence coverage')
        provenance[family] = {'predictions': str(Path(path).resolve()),
            'sha256': hashlib.file_digest(open(path, 'rb'), 'sha256').hexdigest(),
            'confidence': confidence, 'calibration': calibration.get(family), 'calibration_curve': curve}
        images = {im['id']: im for im in data['images']}
        seen = set()
        for image_id, pred in tqdm(prediction_rows(path), total=len(images), desc=family):
            assert image_id not in seen
            seen.add(image_id)
            im = images[image_id]
            values = score(pred, im, data['annotations'][im['id']], data['category_to_index'], args.coordinate_size, confidence)
            targets.append({'image_id': im['id'], 'detector': family, **values})
        assert seen == images.keys(), f'{family}: incomplete predictions'
    if targets:
        pd.DataFrame(targets).to_csv(args.output / 'targets.csv', index=False)
        write_json(args.output / 'targets.json', {'detectors': provenance, 'coordinate_size': args.coordinate_size,
            'target': 'lrp stores 1 - LRP error; higher is better',
            'lrp_match_threshold': .5, 'empty_image_scores': 1.,
            'localization_error_without_TP': 1., 'crowd': 'ignore overlapping crowd predictions lacking a valid non-crowd match'})
    print(rows.groupby('split').agg(images=('image_id', 'size'), groups=('group', 'nunique')))


if __name__ == '__main__':
    main()
