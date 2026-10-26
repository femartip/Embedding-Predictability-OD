"""Profile GPU settings and compare short learning-rate pilots; never launch final training."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.models.catalog import MODELS
from src.data.dataset import ROOT, DEFAULT_COCO, annotation_file, resolve_image


def number(value):
    value = float(value) if value is not None else None
    return value if value is not None and math.isfinite(value) else None


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def histories(family, artifacts_dir=None):
    from src.data.dataset import ROOT
    output = (artifacts_dir or ROOT / 'models') / family
    if family == 'yolo':
        rows = []
        if (output / 'train/results.csv').exists():
            with (output / 'train/results.csv').open() as handle:
                for row in csv.DictReader(handle):
                    if not number(row.get('epoch')):
                        continue
                    mapping = {'epoch': 'epoch', 'time': 'elapsed_seconds',
                        'metrics/mAP50-95(B)': 'val_map', 'metrics/mAP50(B)': 'val_map_50',
                        'metrics/precision(B)': 'val_precision', 'metrics/recall(B)': 'val_recall',
                        'lr/pg0': 'learning_rate'}
                    mapping.update({f'{split}/{name}_loss': f'{split}_{name}_loss'
                                    for split in ['train', 'val'] for name in ['box', 'cls', 'dfl']})
                    rows.append({target: number(row.get(source)) for source, target in mapping.items()})
        return rows
    if family == 'rf-detr':
        rows = []
        for row in read_jsonl(output / 'log.txt'):
            normalized = {'epoch': row.get('epoch', 0) + 1, 'train_loss': number(row.get('train_loss')),
                'val_loss': number(row.get('test_loss')), 'learning_rate': number(row.get('train_lr'))}
            scores = row.get('test_coco_eval_bbox', [])
            for index, key in [(0, 'val_map'), (1, 'val_map_50'), (2, 'val_map_75'), (8, 'val_mar_100')]:
                if len(scores) > index:
                    normalized[key] = number(scores[index])
            extra = row.get('test_results_json', {})
            for item in extra.get('class_map', []):
                if item.get('class') in ['Vehicle', 'VulnerableVehicle', 'Pedestrian']:
                    normalized[f"val_map_{item['class']}"] = number(item.get('map@50:95'))
            for key in ['precision', 'recall']:
                if key in extra:
                    normalized[f'val_{key}'] = number(extra[key])
            rows.append(normalized)
        return rows
    rows = []
    for row in read_jsonl(output / 'learning_history.jsonl'):
        normalized = {key: number(value) for key, value in row.items() if key not in ('time', 'nonfinite_fields')}
        if 'loss' in row:
            normalized['train_loss'] = number(row['loss'])
        for key, value in row.items():
            if key.startswith('eval_'):
                normalized['val_' + key[5:]] = number(value)
        if 'nonfinite_fields' in row:
            normalized['nonfinite_fields'] = row['nonfinite_fields']
        rows.append(normalized)
    return rows



def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def execute(command, log):
    print('Running', log.name, flush=True)
    with log.open('w') as handle:
        process = subprocess.run([sys.executable, '-u', *map(str, command)], cwd=ROOT,
            stdout=handle, stderr=subprocess.STDOUT,
            env={**os.environ, 'WANDB_MODE': 'offline', 'MPLCONFIGDIR': str(ROOT/'logs/tuning/matplotlib'),
                 'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '4'})
    return process.returncode


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models', nargs='+', default=list(MODELS))
    p.add_argument('--coco-dir', type=Path, default=ROOT/'data/tuning/coco')
    p.add_argument('--phase', choices=['profile', 'pilots', 'all'], default='all')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--rates',nargs='+',type=float)
    p.add_argument('--backbone-lr',type=float,default=1e-5)
    p.add_argument('--semantic-init',action='store_true')
    p.add_argument('--compare-existing',action='store_true')
    args = p.parse_args()
    if not list(args.coco_dir.glob('*_train.json')):
        import numpy as np
        args.coco_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(43)
        for split, count in [('train',768),('val',192),('test',0)]:
            data = json.loads(annotation_file(DEFAULT_COCO,split).read_text())
            positions = sorted(rng.choice(len(data['images']),count,replace=False))
            images = [data['images'][i] for i in positions]
            for image in images:
                image['file_name'] = str(resolve_image(image['file_name']))
            ids = {im['id'] for im in images}
            save(args.coco_dir/f'zod_{split}.json', {'images':images,
                'annotations':[a for a in data['annotations'] if a['image_id'] in ids], 'categories':data['categories']})
    output = ROOT/'logs/tuning'
    output.mkdir(parents=True, exist_ok=True)
    settings_file = ROOT/'data/tuning/recipes.json'
    settings = json.loads(settings_file.read_text()) if settings_file.exists() else {}
    status_file = output/'status.json'
    summary = []
    for family in args.models:
        save(status_file, {'family': family, 'phase': 'profiling', 'pid': os.getpid()})
        backend = MODELS[family].backend
        base = 16 if family == 'yolo' else 1 if family == 'rf-detr' else 2
        precisions = ['fp16'] if family == 'yolo' else ['bf16'] if family == 'rf-detr' else ['bf16', 'fp16']
        if args.phase != 'pilots':
            probes = []
            for precision in ['fp32', *precisions]:
                batches = [base] if precision == 'fp32' else [16,32,48,64] if family == 'yolo' else [base,base*2,base*4,base*8]
                for batch in batches:
                    tag = f'{family}-{precision}-b{batch}'
                    path = output/f'{tag}.json'
                    if not path.exists():
                        command = ['src/models/hyperparam_tuning/profile_training.py', family, '--coco-dir', args.coco_dir,
                                   '--batch-size', batch, '--precision', precision, '--output', path]
                        result = execute(command, output/f'{tag}.log')
                        if result:
                            tail = (output/f'{tag}.log').read_text(errors='replace')[-2500:]
                            print(tag, 'failed:', tail[-600:], flush=True)
                            save(output/f'{tag}.failed.json', {'returncode': result, 'tail': tail})
                            break
                    if path.exists():
                        result = json.loads(path.read_text())
                        probes.append(result)
                        print(tag, result['images_per_second'], 'images/s', result['peak_gib'], 'GiB', flush=True)
            eligible = [r for r in probes if r['peak_gib'] < 20.]
            best = max(eligible, key=lambda r: r['images_per_second'])
            effective = 64 if family == 'yolo' else 16
            microbatch = min(best['batch_size'], effective)
            settings[family] = {'batch_size': microbatch, 'amp': best['precision'] != 'fp32',
                'precision': best['precision'] if best['precision'] != 'fp32' else 'bf16',
                'workers': args.workers, 'patience': 5, 'epochs': 50 if family == 'yolo' else 30,
                'profile': best}
            if family != 'yolo':
                settings[family]['grad_accum_steps'] = max(1,effective//microbatch)
            save(settings_file, settings)
        if args.phase == 'profile':
            continue
        save(status_file, {'family': family, 'phase': 'learning-rate pilots', 'pid': os.getpid()})
        recipe = settings[family]
        rates = [.001, .003] if family == 'yolo' else [.001, .005] if backend == 'torchvision' else [5e-5,1e-4] if family == 'detr' else [1e-4,3e-4]
        rates = args.rates or rates
        trials = []
        for lr in rates:
            variant = '-semantic' if args.semantic_init else ''
            parent = ROOT/f'models/tuning/{family}{variant}-lr{lr:g}'
            trial = parent/family
            command = ['src/models/train.py', family, '--coco-dir', args.coco_dir, '--output-dir', trial,
                       '--batch-size', recipe['batch_size'], '--workers', args.workers,
                       '--lr', lr, '--backbone-lr', args.backbone_lr, '--precision', recipe['precision'],
                       '--amp' if recipe['amp'] else '--no-amp', '--warmup-ratio', '.1', '--patience', args.epochs+1]
            if args.semantic_init:
                command += ['--semantic-init']
            if family == 'yolo':
                command += ['--init-checkpoint', ROOT/'models/yolo/before_tuning/last.pt', '--optimizer', 'MuSGD', '--warmup-ratio', '0']
            if family != 'yolo':
                command += ['--grad-accum-steps', recipe['grad_accum_steps']]
            if backend == 'torchvision':
                steps = math.ceil(768/(recipe['batch_size']*recipe['grad_accum_steps']))
                command += ['--max-steps', steps*args.epochs, '--eval-every', steps]
            else:
                command += ['--epochs', args.epochs]
            done = trial/'pilot.json'
            if done.exists():
                result = json.loads(done.read_text())
            else:
                started = time.perf_counter()
                code = execute(command, output/f'{family}{variant}-lr{lr:g}.log')
                if code:
                    print(f'{family} lr={lr} pilot failed; inspect log', flush=True)
                    continue
                history = histories(family, parent)
                val = [r for r in history if r.get('val_map') is not None]
                result = {'family': family, 'lr': lr, 'seconds': time.perf_counter()-started,
                    'semantic_init':args.semantic_init,'backbone_lr':args.backbone_lr,
                    'validation_maps': [r['val_map'] for r in val],
                    'epochs': [r['epoch'] for r in val], 'history': history,
                    'best_map': max(r['val_map'] for r in val),
                    'learning_verified': max(r['val_map'] for r in val[1:]) > val[0]['val_map']}
                save(done, result)
            trials.append(result)
            print(f'{family} lr={lr} AP:', result['validation_maps'], flush=True)
        assert trials, f'{family}: all pilots failed; inspect logs/tuning'
        best = max(trials, key=lambda r: r['best_map'])
        if args.compare_existing and (not best['learning_verified'] or best['best_map']<recipe['pilot']['best_map']):
            print(family,'retaining previous settings; the alternative did not improve validation',flush=True)
            continue
        recipe.update(lr=best['lr'], backbone_lr=args.backbone_lr, pilot=best,semantic_init=args.semantic_init)
        if family == 'yolo':
            recipe.update(init_checkpoint=str(ROOT/'models/yolo/before_tuning/best.pt'), optimizer='MuSGD', epochs=30,warmup_ratio=0.)
        if backend == 'torchvision':
            recipe.pop('epochs')
            recipe.update(max_steps=50000, eval_every=5000)
        save(settings_file, settings)
        summary.extend([{k: v for k, v in r.items() if k not in ['history','epochs']} for r in trials])
        save(output/'pilots.json', summary)
    save(status_file, {'phase': 'finished; final training NOT started', 'recipes': str(settings_file)})
    print('Tuning finished. Final training has NOT been started.', flush=True)


if __name__ == '__main__':
    main()
