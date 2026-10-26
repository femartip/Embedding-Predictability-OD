"""Run assessor experiments and save figures and CSV tables in one results folder."""
import argparse
import subprocess
import sys
from pathlib import Path

import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score

from src.data.dataset import ROOT
from .data import DEFAULT_DATA
from .encoders import ENCODERS


def risk_curve(rows, lower_better):
    ordered = rows.sort_values('predicted', ascending=lower_better)
    risk = ordered.actual if lower_better else 1-ordered.actual
    risk = risk.groupby(ordered.predicted).transform('mean').to_numpy()
    return np.cumsum(risk)/np.arange(1, len(rows)+1)


def metrics(rows, lower_better=False):
    actual, predicted = rows.actual.to_numpy(), rows.predicted.to_numpy()
    rho = spearmanr(actual, predicted).statistic if np.std(predicted) > 1e-8 and np.std(actual) > 1e-8 else np.nan
    return {'mae': mean_absolute_error(actual, predicted), 'r2': r2_score(actual, predicted),
            'spearman': rho, 'aurc': risk_curve(rows, lower_better).mean()}


def interval(rows, repeats, seed):
    groups = [g for _, g in rows.groupby('group')]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        sample = pd.concat([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        values.append(mean_absolute_error(sample.actual, sample.predicted))
    return np.quantile(values, [.025, .975])


def diagnostics(rows, config, output):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    axes[0, 0].scatter(rows.actual, rows.predicted, s=5, alpha=.25)
    axes[0, 0].plot([0, 1], [0, 1], 'k--')
    axes[0, 0].set(xlabel='Measured score', ylabel='Predicted score', xlim=(0, 1), ylim=(0, 1))
    axes[0, 1].hist(rows.predicted-rows.actual, bins=30)
    axes[0, 1].set(xlabel='Prediction error', ylabel='Images')
    lower = config['target'] not in ['iou', 'lrp']
    risk = rows.actual.to_numpy() if lower else 1-rows.actual.to_numpy()
    n = np.arange(1, len(rows)+1)
    axes[1, 0].plot(n/len(rows), risk_curve(rows, lower), label='Assessor')
    axes[1, 0].plot(n/len(rows), np.cumsum(np.sort(risk))/n, label='Oracle')
    axes[1, 0].axhline(risk.mean(), color='gray', linestyle='--', label='Random expectation')
    axes[1, 0].set(xlabel='Coverage (fraction accepted)', ylabel='Mean detection error')
    axes[1, 0].legend()
    axes[1, 1].scatter(rows.objects, abs(rows.predicted-rows.actual), s=5, alpha=.25)
    axes[1, 1].set(xlabel='Ground-truth object count', ylabel='Absolute score error')
    axes[1, 2].scatter(rows.mean_box_height, abs(rows.predicted-rows.actual), s=5, alpha=.25)
    axes[1, 2].set(xlabel='Mean GT box height (source pixels)', ylabel='Absolute score error')
    classes = [c for c in rows if c.startswith('objects_') and (rows[c] > 0).any()]
    axes[0, 2].bar([c.removeprefix('objects_') for c in classes],
                   [abs(rows.predicted-rows.actual)[rows[c] > 0].mean() for c in classes])
    axes[0, 2].set(ylabel='MAE on images containing class')
    fig.tight_layout()
    fig.savefig(output, dpi=130)
    plt.close(fig)


def save_results(runs, data, output, bootstrap=200):
    output.mkdir(parents=True, exist_ok=True)
    images = pd.read_csv(data / 'images.csv').set_index('image_id')
    summary = []
    for file in sorted(runs.glob('*/predictions.csv')):
        config = json.loads(file.with_name('config.json').read_text())
        predictions = pd.read_csv(file)
        run = file.parent.name
        curves = file.with_name('history.csv')
        if curves.exists():
            history = pd.read_csv(curves)
            fig, ax = plt.subplots(1, 2, figsize=(9, 3))
            ax[0].plot(history.epoch, history.train_mse)
            ax[0].set(xlabel='Epoch', ylabel='Training MSE')
            ax[1].plot(history.epoch, history.val_mae)
            ax[1].set(xlabel='Epoch', ylabel='Validation MAE')
            fig.tight_layout()
            fig.savefig(output / f'{run}-learning.png', dpi=130)
            plt.close(fig)
        for (split, detector), group in predictions.groupby(['split', 'detector']):
            values = metrics(group, config['target'] not in ['iou', 'lrp'])
            lo, hi = interval(group, bootstrap, config['seed'])
            name = config['features'] or f'{config["encoder"]}-{config["pool"]}-{config["size"]}-grid{config["grid"]}'
            if config['regressor'] == 'constant':
                name = 'constant'
            method = f'{name}:{config["mode"]}:{config["regressor"]}:{"shared" if config["shared"] else "separate"}'
            summary.append({'run': run, 'method': method, 'detector': detector, 'target': config['target'],
                            'split': split, 'seed': config['seed'], 'fraction': config['fraction'],
                            'trainable_parameters': config['trainable_parameters'],
                            'train_images': config['train_images'], 'images': len(group),
                            'groups': group.group.nunique(), **values, 'mae_ci_low': lo, 'mae_ci_high': hi})
            if split != 'test':
                continue
            diagnostics(group, config, output / f'{run}-{detector}-diagnostics.png')
            error = abs(group.predicted-group.actual)
            examples = group.loc[[error.idxmin(), error.idxmax()]]
            fig, axes = plt.subplots(1, 2, figsize=(10, 3))
            for ax, (_, row) in zip(axes, examples.iterrows()):
                with Image.open(images.loc[row.image_id, 'filepath']) as original:
                    image = original.convert('RGB')
                image.thumbnail((700, 400))
                ax.imshow(image)
                ax.set_title(f'Image {row.image_id}: measured {row.actual:.3f}, predicted {row.predicted:.3f}')
                ax.axis('off')
            fig.tight_layout()
            fig.savefig(output / f'{run}-{detector}-examples.png', dpi=130)
            plt.close(fig)
    table = pd.DataFrame(summary)
    table.to_csv(output / 'summary.csv', index=False)
    aggregate = table.groupby(['method', 'detector', 'target', 'split', 'fraction'], as_index=False).agg(
        mae_mean=('mae', 'mean'), mae_seed_std=('mae', 'std'), runs=('seed', 'size'),
        spearman_mean=('spearman', 'mean'), aurc_mean=('aurc', 'mean'))
    aggregate.to_csv(output / 'aggregate.csv', index=False)
    validation = table[table.split == 'val']
    for (target, detector), rows in validation.groupby(['target', 'detector']):
        fig, ax = plt.subplots(figsize=(10, 5))
        for method, group in rows.groupby('method'):
            curve = group.groupby('train_images').mae.agg(['mean', 'std']).sort_index()
            ax.errorbar(curve.index, curve['mean'], yerr=curve['std'].fillna(0), marker='o', label=method)
        ax.set(xlabel='Training images', ylabel='Validation MAE', title=f'{detector}: {target}')
        ax.legend(fontsize=6, bbox_to_anchor=(1.02, 1), loc='upper left')
        fig.tight_layout()
        fig.savefig(output / f'label-budget-{target}-{detector}.png', dpi=130)
        plt.close(fig)
    print(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=DEFAULT_DATA)
    p.add_argument('--features', type=Path, help='Directory produced by features.py')
    p.add_argument('--detectors', nargs='+')
    p.add_argument('--encoders', nargs='+', choices=ENCODERS, default=ENCODERS)
    p.add_argument('--targets', nargs='+', default=['iou', 'lrp'])
    p.add_argument('--pools', nargs='+', default=['global', 'mean', 'spatial'])
    p.add_argument('--seeds', nargs='+', default=['43'])
    p.add_argument('--fractions', nargs='+', default=['1'])
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--device', default='cuda')
    p.add_argument('--shared', action='store_true')
    p.add_argument('--output', type=Path, default=ROOT / 'models/assessors')
    p.add_argument('--results', type=Path, default=ROOT / 'results/assessors')
    p.add_argument('--bootstrap', type=int, default=200)
    p.add_argument('--report-only', action='store_true', help='Analyze saved runs without training')
    args = p.parse_args()
    if args.report_only:
        save_results(args.output, args.data, args.results, args.bootstrap)
        return
    if args.features is None or not args.detectors:
        p.error('--features and --detectors are required unless --report-only is used')
    def execute(module, *flags):
        subprocess.run([sys.executable, '-m', 'src.assessors.'+module, *map(str, flags)], check=True, cwd=ROOT)
    for target in args.targets:
        common = ['--data', args.data, '--detectors', *args.detectors, '--target', target,
                  '--seeds', *args.seeds, '--fractions', *args.fractions,
                  '--epochs', args.epochs, '--device', args.device, '--output', args.output]
        if args.shared:
            common += ['--shared']
        execute('train', *common, '--regressor', 'constant')
        paths = [args.features / 'quality.npz'] + [args.features / f'{e}-{pool}.npz' for e in args.encoders for pool in args.pools]
        for path in paths:
            for regressor in ['ridge', 'mlp']:
                execute('train', *common, '--features', path, '--regressor', regressor)
    save_results(args.output, args.data, args.results, args.bootstrap)


if __name__ == '__main__':
    main()
