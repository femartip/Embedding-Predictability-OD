"""Plot short detector pilots separately from full-data fine-tuning results."""
import sys
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.models.catalog import MODELS
from src.data.dataset import ROOT


def main():
    output = ROOT/'reports/tuning'
    output.mkdir(parents=True, exist_ok=True)
    recipes = json.loads((ROOT/'data/tuning/recipes.json').read_text())
    profiles = [json.loads(p.read_text()) for p in (ROOT/'logs/tuning').glob('*.json')
                if 'seconds_per_batch' in json.loads(p.read_text())]
    for row in profiles:
        row.setdefault('rf_groups',13 if row['family']=='rf-detr' else None)
    profile_table = pd.DataFrame([{k: v for k,v in row.items() if k not in ['losses','seconds_per_batch']} for row in profiles])
    profile_table.to_csv(output/'throughput.csv', index=False)
    summaries, sections = [], []
    for family in MODELS:
        trials = [json.loads(p.read_text()) for p in (ROOT/'models/tuning').glob(f'{family}*/{family}/pilot.json')]
        if not trials:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
        for trial in trials:
            label = f'LR {trial["lr"]:g}'
            if 'rf_groups' in trial:
                label += f' / groups {trial["rf_groups"]}'
            if trial.get('semantic_init'):
                label += ' / COCO head init'
            if trial.get('channels_last'):
                label += ' / NHWC'
            rows = trial['history']
            losses = [(r['epoch'], r.get('train_loss', r.get('train_cls_loss'))) for r in rows
                      if r.get('epoch') is not None and r.get('train_runtime') is None
                      and (r.get('train_loss') is not None or r.get('train_cls_loss') is not None)]
            if losses:
                x,y = zip(*losses)
                axes[0].plot(x,y,label=label)
            axes[1].plot(trial['epochs'], trial['validation_maps'], marker='o',label=label)
            summaries.append({'family':family,'lr':trial['lr'],'first_map':trial['validation_maps'][0],
                'last_map':trial['validation_maps'][-1],'best_map':trial['best_map'],
                'learning_verified':trial['learning_verified'],'pilot_seconds':trial['seconds'],
                'semantic_init':trial.get('semantic_init',False),
                'channels_last':trial.get('channels_last',False),
                'rf_groups':trial.get('rf_groups',13 if family == 'rf-detr' else None)})
        subset = profile_table[profile_table.family == family]
        for (precision, groups, channels), group in subset.groupby(['precision','rf_groups','channels_last'],dropna=False):
            group = group.sort_values('batch_size')
            label = f'{precision} / g{int(groups)}' if family=='rf-detr' else precision
            if channels:
                label += ' / NHWC'
            axes[2].plot(group.batch_size,group.images_per_second,marker='o',label=label)
        axes[0].set(xlabel='Epoch',ylabel='Training loss (YOLO: classification)')
        axes[1].set(xlabel='Epoch',ylabel='Validation AP50–95')
        axes[2].set(xlabel='Microbatch',ylabel='GPU probe images/second')
        for ax in axes:
            ax.legend()
            ax.grid(alpha=.2)
        fig.suptitle(family)
        fig.tight_layout()
        fig.savefig(output/f'{family}.png',dpi=150)
        plt.close(fig)
        sections.append(f'<h2>{html.escape(family)}</h2><img width="1200" src="{family}.png">')
    table = pd.DataFrame(summaries)
    table.to_csv(output/'pilots.csv',index=False)
    chosen = pd.DataFrame([{'family':f, **{k:v for k,v in r.items() if k not in ['profile','pilot']}} for f,r in recipes.items()])
    chosen.to_csv(output/'selected_settings.csv',index=False)
    body = '<!doctype html><meta charset="utf-8"><title>Detector tuning pilots</title><h1>Detector tuning pilots</h1>'
    body += '<p>768 training and 192 validation images, seed 43. Test images are excluded. '
    body += 'YOLO starts from its preserved ZOD checkpoint; other families start from COCO. '
    body += 'GPU probes exclude loading/augmentation and use SGD; pilot timings include setup and validation. '
    body += 'These short pilots establish learning signals, not final accuracy or full-data training speed. '
    body += 'Final training has not been started.</p><h2>Selected settings</h2>'+chosen.to_html(index=False)
    body += '<h2>Pilot validation measurements</h2>'+table.to_html(index=False)
    body += ''.join(sections)+'<h2>GPU throughput probes</h2>'+profile_table.to_html(index=False)
    (output/'index.html').write_text(body)
    print(output/'index.html')


if __name__ == '__main__':
    main()
