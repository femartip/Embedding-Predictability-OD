"""Create a grouped image split and rescore frozen detector predictions."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torchvision.ops import box_iou, box_area
from tqdm import tqdm

from src.data.dataset import DEFAULT_COCO, load_coco, clipped_annotations, validate_splits
from src.models.metrics import get_metrics, LocalizationRecallPrecision
from .data import DEFAULT_DATA, write_json


def payload(path):
    data = json.loads(Path(path).read_text())
    return {int(k): v for k, v in data.items()}


def score(pred, image, annotations, categories, size, confidence):
    sx, sy = size[0] / image['width'], size[1] / image['height']
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
    values = get_metrics([p], [target], ['iou', 'lrp'])
    loc, fp, fn, tp = LocalizationRecallPrecision()._process_single(p, target)
    return values | {'fp_rate': float(fp / (tp+fp)) if tp+fp else 0.,
                     'fn_rate': float(fn / (tp+fn)) if tp+fn else 0.,
                     'localization_error': float(loc / tp) if tp else 1.,
                     'fp': float(fp), 'fn': float(fn), 'tp': float(tp)}


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
    args = p.parse_args()
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
    targets = []
    provenance = {}
    for entry in args.predictions:
        family, path = entry.split('=', 1)
        preds = payload(path)
        assert set(rows.image_id) <= preds.keys(), f'{family}: incomplete predictions'
        confidence, curve = args.confidence, {}
        if family in calibration:
            val = load_coco(args.coco_dir, 'val', args.image_root)
            vp = payload(calibration[family])
            for threshold in args.thresholds:
                curve[str(threshold)] = np.mean([score(vp[im['id']], im, val['annotations'][im['id']],
                    val['category_to_index'], args.coordinate_size, threshold)['lrp'] for im in val['images']])
            confidence = float(max(curve, key=curve.get))
        for prediction_path, expected_split in [(path, 'test')] + ([(calibration[family], 'val')] if family in calibration else []):
            meta_path = Path(prediction_path).with_name('predictions.meta.json')
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                assert meta['split'] == expected_split and not meta['pretrained'] and not meta['partial']
                assert meta['coordinate_size'] == args.coordinate_size and meta['classes'] == data['classes']
                minimum = min(args.thresholds) if expected_split == 'val' else confidence
                assert meta['confidence_threshold'] <= minimum, 'Regenerate predictions at a lower confidence floor'
            else:
                print(f'{prediction_path}: legacy predictions; assuming declared coordinates, class order and confidence coverage')
        provenance[family] = {'predictions': str(Path(path).resolve()),
            'sha256': hashlib.file_digest(open(path, 'rb'), 'sha256').hexdigest(),
            'confidence': confidence, 'calibration': calibration.get(family), 'calibration_curve': curve}
        for im in tqdm(data['images'], desc=family):
            values = score(preds[im['id']], im, data['annotations'][im['id']], data['category_to_index'], args.coordinate_size, confidence)
            targets.append({'image_id': im['id'], 'detector': family, **values})
    if targets:
        pd.DataFrame(targets).to_csv(args.output / 'targets.csv', index=False)
        write_json(args.output / 'targets.json', {'detectors': provenance, 'coordinate_size': args.coordinate_size,
            'iou_match': 'positive overlap, class-aware greedy matching; FP/FN zeros',
            'lrp_match_threshold': .5, 'empty_image_scores': 1.,
            'localization_error_without_TP': 1., 'crowd': 'ignore overlapping crowd predictions lacking a valid non-crowd match'})
    print(rows.groupby('split').agg(images=('image_id', 'size'), groups=('group', 'nunique')))


if __name__ == '__main__':
    main()
