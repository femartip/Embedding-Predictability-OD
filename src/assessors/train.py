"""Fit frozen-feature regressors or fine-tune an image assessor."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.dataset import ROOT
from .data import DEFAULT_DATA, Images, ImageBatch, read_data, write_json
from .encoders import ENCODERS, Encoder, head


def to_device(x, device):
    return {k: v.to(device) for k, v in x.items()} if isinstance(x, dict) else x.to(device)


def predict(model, loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for x, _ in loader:
            predictions.append(model(to_device(x, device)).cpu().numpy())
    return np.concatenate(predictions)


class Assessor(nn.Module):
    def __init__(self, encoder, pool, outputs):
        super().__init__()
        self.encoder, self.pool = encoder, pool
        self.head = head(encoder.dim * (np.prod(encoder.grid) if pool == 'spatial' else 1), outputs)

    def forward(self, x):
        return self.head(self.encoder(x, self.pool))


def fit_neural(model, loaders, args, output):
    model.to(args.device)
    groups = [{'params': model.head.parameters(), 'lr': args.lr}] if isinstance(model, Assessor) else [{'params': model.parameters(), 'lr': args.lr}]
    if isinstance(model, Assessor):
        groups.append({'params': [p for p in model.encoder.parameters() if p.requires_grad], 'lr': args.encoder_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    best, stale, history = float('inf'), 0, []
    for epoch in range(args.epochs):
        model.train()
        if isinstance(model, Assessor):
            for module in model.encoder.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        total, count = 0., 0
        for x, y in loaders['train']:
            x, y = to_device(x, args.device), y.to(args.device)
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item()*len(y)
            count += len(y)
        values = predict(model, loaders['val'], args.device)
        truth = np.concatenate([y.numpy() for _, y in loaders['val']])
        error = mean_absolute_error(truth, values)
        history.append({'epoch': epoch+1, 'train_mse': total/count, 'val_mae': error})
        pd.DataFrame(history).to_csv(output / 'history.csv', index=False)
        print(output.name, history[-1], flush=True)
        if error < best:
            best, stale = error, 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= args.patience:
            break
    model.load_state_dict(state)
    torch.save(state, output / 'best.pt')
    return {split: predict(model, loader, args.device) for split, loader in loaders.items() if split != 'train'}


def run(args, detectors, seed, fraction):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rows, targets = read_data(args.data)
    labels = targets.pivot(index='image_id', columns='detector', values=args.target)[detectors]
    labels.columns = ['y_'+d for d in detectors]
    rows = rows.merge(labels, left_on='image_id', right_index=True, validate='one_to_one')
    assert rows.image_id.nunique() == len(pd.read_csv(args.data / 'images.csv')) and rows[labels.columns].notna().all().all()
    train = np.flatnonzero(rows.split == 'train')
    keep = np.random.default_rng(seed).permutation(train)[:max(2, int(len(train)*fraction))]
    rows = rows.loc[(rows.split != 'train') | rows.index.isin(keep)].reset_index(drop=True)
    features, encoder_config = None, None
    if args.features:
        archive = np.load(args.features)
        indexed = pd.DataFrame(archive['features'], index=archive['image_id'])
        assert indexed.index.is_unique
        features = indexed.loc[rows.image_id].to_numpy(dtype=np.float32)
        metadata = args.features.with_suffix('.json')
        encoder_config = json.loads(metadata.read_text()) if metadata.exists() else None
    name = f'{args.features.parent.name}-{args.features.stem}' if args.features else f'{args.encoder}-{args.pool}-{args.size[0]}x{args.size[1]}-grid{args.grid[0]}x{args.grid[1]}'
    if args.regressor == 'constant':
        name = 'constant'
    tag = f'{name}-{args.mode}-{args.regressor}-{args.target}-{"+".join(detectors)}-s{seed}-n{fraction:g}'
    output = args.output / tag
    output.mkdir(parents=True)
    indices = {split: np.flatnonzero(rows.split == split) for split in ['train', 'val', 'test']}
    y = rows[labels.columns].to_numpy(dtype=np.float32)
    if args.regressor in ['constant', 'ridge']:
        if args.regressor == 'constant':
            mean = y[indices['train']].mean(0)
            predictions = {s: np.broadcast_to(mean, (len(indices[s]), len(detectors))) for s in ['val', 'test']}
            joblib.dump(mean, output / 'regressor.pkl')
        else:
            candidates = []
            for alpha in [.1, 1., 10., 100.]:
                model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver='lsqr'))
                model.fit(features[indices['train']], y[indices['train']])
                error = mean_absolute_error(y[indices['val']], model.predict(features[indices['val']]).clip(0, 1))
                candidates.append((error, alpha, model))
            _, alpha, model = min(candidates, key=lambda t: t[0])
            predictions = {s: model.predict(features[indices[s]]).clip(0, 1).reshape(-1, len(detectors)) for s in ['val', 'test']}
            joblib.dump(model, output / 'regressor.pkl')
            write_json(output / 'selection.json', {'alpha': alpha, 'validation_mae': [float(c[0]) for c in candidates]})
    else:
        if features is not None:
            model = head(features.shape[1], len(detectors))
            datasets = {s: TensorDataset(torch.from_numpy(features[i]), torch.from_numpy(y[i])) for s, i in indices.items()}
            collate = None
        else:
            encoder = Encoder(args.encoder, args.size, args.grid, args.revision)
            encoder.tune(args.mode)
            encoder_config = encoder.config()
            model = Assessor(encoder, args.pool, len(detectors))
            datasets = {s: Images(rows.iloc[i]) for s, i in indices.items()}
            collate = ImageBatch(encoder, rows, labels.columns)
        loaders = {s: DataLoader(dataset, batch_size=args.batch_size, shuffle=s == 'train', collate_fn=collate)
                   for s, dataset in datasets.items()}
        predictions = fit_neural(model, loaders, args, output)
    parts = []
    for split, values in predictions.items():
        selected = rows.iloc[indices[split]]
        for j, detector in enumerate(detectors):
            columns = ['image_id', 'group', 'split', 'objects', 'mean_box_height'] + [c for c in selected if c.startswith('objects_')]
            parts.append(selected[columns].assign(
                detector=detector, actual=y[indices[split], j], predicted=values[:, j]))
    pd.concat(parts).to_csv(output / 'predictions.csv', index=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(detectors=detectors, seed=seed, fraction=fraction, encoder_config=encoder_config,
                  train_images=len(indices['train']),
                  feature_dimension=features.shape[1] if features is not None else None,
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad) if args.regressor == 'mlp' else None,
                  split_sha256=hashlib.file_digest(open(args.data / 'images.csv', 'rb'), 'sha256').hexdigest(),
                  target_sha256=hashlib.file_digest(open(args.data / 'targets.csv', 'rb'), 'sha256').hexdigest())
    write_json(output / 'config.json', config)
    print(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=DEFAULT_DATA)
    source = p.add_mutually_exclusive_group()
    source.add_argument('--features', type=Path)
    source.add_argument('--encoder', choices=ENCODERS)
    p.add_argument('--detectors', nargs='+', required=True)
    p.add_argument('--shared', action='store_true')
    p.add_argument('--target', choices=['iou', 'lrp', 'fp_rate', 'fn_rate', 'localization_error'], default='iou')
    p.add_argument('--regressor', choices=['constant', 'ridge', 'mlp'], default='mlp')
    p.add_argument('--mode', choices=['frozen', 'partial', 'full'], default='frozen')
    p.add_argument('--pool', choices=['global', 'mean', 'spatial'], default='global')
    p.add_argument('--size', nargs=2, type=int, default=[224, 448])
    p.add_argument('--grid', nargs=2, type=int, default=[2, 4])
    p.add_argument('--revision', default='main')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--encoder-lr', type=float, default=1e-5)
    p.add_argument('--device', default='cuda')
    p.add_argument('--seeds', nargs='+', type=int, default=[43])
    p.add_argument('--fractions', nargs='+', type=float, default=[1.])
    p.add_argument('--output', type=Path, default=ROOT / 'models/assessors')
    args = p.parse_args()
    assert args.regressor == 'constant' or args.features or args.encoder
    assert args.regressor != 'ridge' or args.features
    assert not args.features or args.mode == 'frozen'
    assert not args.encoder or (args.mode != 'frozen' and args.regressor == 'mlp')
    assert all(0 < f <= 1 for f in args.fractions)
    torch.set_num_threads(4)
    for detectors in [args.detectors] if args.shared else [[d] for d in args.detectors]:
        for seed in args.seeds:
            for fraction in args.fractions:
                run(args, detectors, seed, fraction)


if __name__ == '__main__':
    main()
