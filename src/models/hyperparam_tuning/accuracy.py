"""Audit frozen detectors, compare resolutions, and run short recovery pilots."""
import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import ROOT, DEFAULT_COCO, annotation_file, resolve_image, load_coco, clipped_annotations
from src.models.catalog import MODELS
from src.models.adapters import batch_predictor as predictor
from src.models.train import detection_metric, save_previews, collate_images
from src.models.hyperparam_tuning.tune import histories

OUTPUT = ROOT / 'results/detector_accuracy'
DATA = ROOT / 'data/tuning/accuracy'
RUNS = ROOT / 'models/tuning/accuracy'
LOGS = ROOT / 'logs/tuning/accuracy'


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def prepare():
    DATA.mkdir(parents=True, exist_ok=True)
    for split, count in [('train', 2048), ('val', 512), ('test', 0)]:
        path = DATA / f'zod_{split}.json'
        if path.exists():
            continue
        data = json.loads(annotation_file(DEFAULT_COCO, split).read_text())
        rng = np.random.default_rng(89)
        selected = sorted(rng.choice(len(data['images']), count, replace=False))
        data['images'] = [data['images'][i] for i in selected]
        for image in data['images']:
            image['file_name'] = str(resolve_image(image['file_name']))
        ids = {im['id'] for im in data['images']}
        data['annotations'] = [a for a in data['annotations'] if a['image_id'] in ids]
        write(path, data)


class ValidationImages(Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        with Image.open(self.images[index]['filepath']) as image:
            return pil_to_tensor(image.convert('RGB')), index


def audit(family, artifact, resolution, tag, coco_dir=DATA, output=OUTPUT, batch_size=1, workers=0):
    path = output / f'{tag}.json'
    if path.exists():
        return
    data = load_coco(coco_dir, 'val')
    predict = predictor(family, artifact, resolution)
    native, reference = detection_metric(), detection_metric()
    previews, image_rows, invalid = [], [], []
    started = time.perf_counter()
    loader = DataLoader(ValidationImages(data['images']), batch_size=batch_size,
        num_workers=workers, collate_fn=collate_images, prefetch_factor=1 if workers else None)
    predictions_path = output/f'{tag}-predictions.jsonl'
    with predictions_path.open('w') as raw, torch.inference_mode():
        for tensors, indices in tqdm(loader, desc=tag):
            images = [Image.fromarray(im.permute(1,2,0).numpy()) for im in tensors]
            batch = predict(list(images))
            assert len(batch) == len(images)
            for image, index, pred in zip(images, indices, batch):
                im = data['images'][index]
                assert image.size == (im['width'], im['height']), im['id']
                pred = {k: v.detach().cpu().float() if k != 'labels' else v.detach().cpu().long() for k,v in pred.items()}
                assert all(torch.isfinite(v).all() for v in pred.values())
                assert set(pred['labels'].tolist()) <= {0,1,2}
                bad = (pred['boxes'][:,2:] < pred['boxes'][:,:2]).any(1)
                for box, score in zip(pred['boxes'][bad], pred['scores'][bad]):
                    invalid.append({'image_id':im['id'],'score':float(score),'box':box.tolist()})
                # Keep invalid predictions as zero-area false positives, never turn them into valid boxes.
                pred['boxes'][:,2:] = torch.maximum(pred['boxes'][:,2:], pred['boxes'][:,:2])
                anns = clipped_annotations(im, data['annotations'][im['id']])
                boxes = torch.tensor([[x,y,x+w,y+h] for x,y,w,h in [a['bbox'] for a in anns]], dtype=torch.float32).reshape(-1,4)
                target = {'boxes': boxes, 'labels': torch.tensor([data['category_to_index'][a['category_id']] for a in anns]).long(),
                    'iscrowd': torch.tensor([a.get('iscrowd',0) for a in anns]), 'area': torch.tensor([a['area'] for a in anns])}
                native.update([pred], [target])
                scale = 1344/im['width']
                reference.update([{**pred, 'boxes': pred['boxes']*scale}],
                                 [{**target, 'boxes': boxes*scale, 'area': target['area']*scale**2}])
                input_scale = resolution/im['height']
                if family == 'yolo':
                    input_scale = resolution/max(im['width'], im['height'])
                elif MODELS[family].backend == 'torchvision':
                    input_scale = min(800/min(im['width'],im['height']),1333/max(im['width'],im['height']))
                elif family == 'detr':
                    input_scale = min(resolution/min(im['width'],im['height']),resolution*864/512/max(im['width'],im['height']))
                for ann in anns:
                    w,h=ann['bbox'][2:]
                    image_rows.append({'image_id':im['id'],'class':data['classes'][data['category_to_index'][ann['category_id']]],
                        'source_height':h, 'reference_area':w*h*scale**2,
                        'input_height':h*input_scale})
                if index < 12:
                    previews.append({'image_id':im['id'], 'filepath':im['filepath'],
                        'target': {k:target[k].tolist() for k in ['boxes','labels']},
                        'prediction': {k:v.tolist() for k,v in pred.items()}})
                raw.write(json.dumps({'image_id':im['id'], **{k:v.tolist() for k,v in pred.items()}})+'\n')
            processed = indices[-1]+1
            if processed % 128 == 0 or processed == len(data['images']):
                write(output/f'{tag}-progress.json', {'processed':processed, 'total':len(data['images']),
                    'elapsed_seconds':time.perf_counter()-started})
    result, ref = native.compute(), reference.compute()
    scores = {k:float(result[k]) for k in ['map','map_50','map_75','map_small','map_medium','map_large','mar_100']}
    scores.update({k+'_reference1344':float(ref[k]) for k in ['map_small','map_medium','map_large']})
    scores.update({data['classes'][int(k)]:float(v) for k,v in zip(result['classes'],result['map_per_class'])})
    if invalid:
        pd.DataFrame(invalid).to_csv(output/f'{tag}-invalid.csv',index=False)
    pd.DataFrame(image_rows).to_csv(output/f'{tag}-objects.csv',index=False)
    save_previews(output/tag, previews, data['classes'], {'audit':tag}, confidence=.3)
    write(path, {'family':family,'artifact':str(artifact),'resolution':resolution,'images':len(data['images']),
        'seconds':time.perf_counter()-started,'scores':scores,'coordinate_space':'original source pixels',
        'size_bins':'COCO native areas and separately uniformly resized to width 1344',
        'confidence_floor':.001,'dtype':'FP32','batch_size':batch_size,
        'split':'val','coco_dir':str(coco_dir),'prediction_file':predictions_path.name,
        'invalid_predictions':len(invalid),'invalid_handling':'zero-area false positives'})
    del predict, native, reference
    gc.collect(); torch.cuda.empty_cache()


def pilots():
    plans=[]
    for family,resolutions in [('yolo',[512,768]),('rf-detr',[560,840])]:
        for resolution in resolutions:
            tag=f'{family}-r{resolution}'
            manifest=json.loads((ROOT/f'models/tuned/{family}/manifest.json').read_text())
            checkpoint=ROOT/f'models/tuned/{family}'/manifest['checkpoint']
            flags=['--resolution',resolution,'--epochs',2,'--init-checkpoint',checkpoint,'--lr',.0001 if family=='rf-detr' else .0003]
            if family=='yolo':
                flags+=['--batch-size',8,'--optimizer','MuSGD','--multi-scale',0,'--precision','fp16','--channels-last','--warmup-ratio',0]
            else:
                flags+=['--batch-size',4,'--grad-accum-steps',4,'--rf-groups',1,'--backbone-lr',1e-6,'--precision','bf16']
            plans.append((family,tag,flags))
    for family in ['d-fine','rt-detr','detr']:
        for backbone in [1e-6,1e-5]:
            tag=f'{family}-bb{backbone:g}'
            flags=['--init-checkpoint',ROOT/f'models/tuned/{family}/best','--resolution',512,'--epochs',3,
                   '--batch-size',8,'--grad-accum-steps',2,'--lr',3e-5,'--backbone-lr',backbone,
                   '--precision','fp16' if family=='detr' else 'bf16','--channels-last']
            plans.append((family,tag,flags))
    plans.append(('d-fine','d-fine-r768-augmented',[
        '--init-checkpoint',ROOT/'models/tuned/d-fine/best','--resolution',768,'--epochs',2,
        '--batch-size',4,'--grad-accum-steps',4,'--lr',3e-5,'--backbone-lr',1e-6,
        '--precision','bf16','--channels-last','--augment']))
    for family,tag,flags in plans:
        parent=RUNS/tag; trial=parent/family
        if (trial/'manifest.json').exists():
            continue
        command=[sys.executable,'-u',str(ROOT/'src/models/train.py'),family,'--coco-dir',str(DATA),
                 '--output-dir',str(trial),'--workers','4','--patience','5',*map(str,flags)]
        write(LOGS/'status.json',{'phase':'pilot','tag':tag,'command':command})
        with (LOGS/f'{tag}.log').open('w') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        audit(family,trial,int(flags[flags.index('--resolution')+1]),tag+'-pilot')


def summarize():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[]
    for path in OUTPUT.glob('*.json'):
        if path.name.endswith('-predictions.json'):
            continue
        result=json.loads(path.read_text())
        if 'scores' in result:
            rows.append({'run':path.stem,'family':result['family'],'resolution':result['resolution'],
                         'artifact':result['artifact'],'precision':result['dtype'],
                         'images':result['images'],'seconds':result['seconds'],**result['scores']})
    table=pd.DataFrame(rows)
    table.to_csv(OUTPUT/'validation_summary.csv',index=False)
    if len(table):
        best=table[table.precision=='FP32'].sort_values('map',ascending=False).drop_duplicates('family')
        best.to_csv(OUTPUT/'best_observed.csv',index=False)
    visibility=[]
    for path in OUTPUT.glob('*-frozen-objects.csv'):
        objects=pd.read_csv(path)
        for name,group in objects.groupby('class'):
            visibility.append({'run':path.stem.removesuffix('-objects'),'class':name,'objects':len(group),
                'median_source_height':group.source_height.median(),
                'median_input_height':group.input_height.median(),
                'fraction_below_12_input_pixels':(group.input_height<12).mean()})
    pd.DataFrame(visibility).to_csv(OUTPUT/'object_visibility.csv',index=False)
    precision=[]
    for row in rows:
        if row['run'].endswith('-frozen'):
            previous=next((r for r in rows if r['run']==row['run'].replace('-frozen','-amp')),None)
            if previous:
                precision.append({'family':row['family'],'resolution':row['resolution'],
                    'amp_dtype':'fp16' if row['family']=='yolo' else 'bf16',
                    'amp_map':previous['map'],'fp32_map':row['map'],
                    'difference_pp':100*(row['map']-previous['map'])})
    precision_table=pd.DataFrame(precision)
    precision_table.to_csv(OUTPUT/'precision_comparison.csv',index=False)
    if len(precision_table):
        original=precision_table[precision_table.resolution.isin([512,560,800])]
        ax=original.set_index('family')[['amp_map','fp32_map']].rename(
            columns={'amp_map':'BF16 (YOLO: FP16)','fp32_map':'FP32'}).plot.bar(figsize=(10,4))
        ax.set(xlabel='',ylabel='Validation AP50–95',title='Same weights and images: evaluation precision')
        ax.figure.tight_layout();ax.figure.savefig(OUTPUT/'evaluation_precision.png',dpi=140);plt.close(ax.figure)
    if len(table):
        columns=['map_small_reference1344','map_medium_reference1344','map_large_reference1344']
        defaults={f:f'{f}-r{560 if f=="rf-detr" else 800 if MODELS[f].backend=="torchvision" else 512}-frozen' for f in MODELS}
        original=table[table.run.isin(defaults.values())]
        ax=original.set_index('family')[columns].rename(columns=dict(zip(columns,['Small','Medium','Large']))).replace(-1,np.nan).plot.bar(figsize=(11,5))
        ax.set(ylabel='Validation AP50–95',title='Frozen panel: size bins at reference width 1344')
        ax.figure.tight_layout();ax.figure.savefig(OUTPUT/'ap_by_size.png',dpi=140);plt.close(ax.figure)
        fig,axes=plt.subplots(2,3,figsize=(12,7))
        for ax,family in zip(axes.flat,['yolo','rf-detr','d-fine','rt-detr','detr']):
            group=table[(table.family==family)&table.run.str.endswith('-frozen')].sort_values('resolution')
            ax.plot(group.resolution,group['map'],marker='o',label='Original checkpoint')
            adapted=table[(table.family==family)&table.run.str.match(family+r'-r\d+.*-pilot')].sort_values('resolution')
            if len(adapted):
                label='2 epochs + augmentation' if family=='d-fine' else '2-epoch continuation'
                ax.plot(adapted.resolution,adapted['map'],marker='x',linestyle='--',label=label)
                ax.legend(fontsize=7)
            label='Longest edge (pixels)' if family=='yolo' else 'Requested short edge (pixels)' if family=='detr' else 'Square side (pixels)'
            ax.set(title=family,xlabel=label,ylabel='Validation AP50–95')
            ax.grid(alpha=.2)
        axes.flat[-1].axis('off')
        fig.suptitle('Validation AP versus inference size')
        fig.tight_layout();fig.savefig(OUTPUT/'inference_resolution.png',dpi=140);plt.close(fig)
    curves=[]
    for path in RUNS.glob('*/*/manifest.json'):
        for row in histories(path.parent.name,path.parent.parent):
            curves.append({'run':path.parent.parent.name,'family':path.parent.name,**row})
    pd.DataFrame(curves).to_csv(OUTPUT/'pilot_history.csv',index=False)
    for family in MODELS:
        fig,axes=plt.subplots(1,3,figsize=(15,3.5))
        original=histories(family,ROOT/'models/tuned')
        validation=[r for r in original if r.get('val_map') is not None]
        axes[0].plot([r['epoch'] for r in validation],[r['val_map'] for r in validation],marker='o')
        axes[0].set(title='Completed full-data run',xlabel='Epoch',ylabel='Validation AP50–95')
        for run in sorted({r['run'] for r in curves if r['family']==family}):
            selected=[r for r in curves if r['run']==run and r.get('val_map') is not None]
            axes[2].plot([r['epoch'] for r in selected],[r['val_map'] for r in selected],marker='o',label=run)
            losses=[r for r in curves if r['run']==run and r.get('train_runtime') is None
                    and r.get('train_loss',r.get('train_cls_loss')) is not None]
            axes[1].plot([r['epoch'] for r in losses],[r.get('train_loss',r.get('train_cls_loss')) for r in losses],label=run)
        axes[1].set(title='Recovery pilot losses',xlabel='Epoch',ylabel='Loss (YOLO: classification)')
        axes[2].set(title='Recovery pilots (different validation sample)',xlabel='Epoch',ylabel='Validation AP50–95')
        if axes[2].lines:
            axes[2].legend(fontsize=7)
        for ax in axes:
            ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(OUTPUT/f'{family}-learning.png',dpi=140);plt.close(fig)
    for family,low,high,labels in [
        ('yolo','yolo-r512-frozen','yolo-r1280-frozen',['Ground truth','512: original weights','1280: same weights']),
        ('rf-detr','rf-detr-r560-frozen','rf-detr-r840-pilot',['Ground truth','560: original weights','840: two-epoch continuation'])]:
        if not (OUTPUT/high/'learning_previews/metadata.json').exists():
            continue
        fig,axes=plt.subplots(2,3,figsize=(15,6))
        for row,index in enumerate([2,6]):
            a=Image.open(OUTPUT/low/f'learning_previews/sample-{index}.jpg')
            b=Image.open(OUTPUT/high/f'learning_previews/sample-{index}.jpg')
            width=a.width//2
            panels=[a.crop((0,34,width,a.height)),a.crop((width,34,a.width,a.height)),b.crop((width,34,b.width,b.height))]
            for ax,panel,label in zip(axes[row],panels,labels):
                ax.imshow(panel);ax.axis('off');ax.set_title(label)
            a.close();b.close()
        fig.suptitle(f'{family}: validation examples, prediction score >= 0.3')
        fig.tight_layout();fig.savefig(OUTPUT/f'{family}-examples.png',dpi=140);plt.close(fig)


def full_validation(batch_size, workers):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output = OUTPUT/'full_validation'
    output.mkdir(parents=True, exist_ok=True)
    selection = output/'selection.csv'
    if not selection.exists():
        pd.read_csv(OUTPUT/'best_observed.csv').to_csv(selection, index=False)
    chosen = pd.read_csv(selection).set_index('family')
    total = len(json.loads(annotation_file(DEFAULT_COCO,'val').read_text())['images'])
    state = {'pid':os.getpid(), 'phase':'validation', 'split':'val', 'images_per_model':total,
             'batch_size':batch_size, 'workers':workers, 'precision':'FP32',
             'models':{f:'pending' for f in MODELS}, 'training_started':False}
    rows = []
    for family in MODELS:
        row = chosen.loc[family]
        state.update(current_model=family)
        state['models'][family] = 'running'
        write(output/'status.json', state)
        audit(family, Path(row.artifact), int(row.resolution), family,
              coco_dir=DEFAULT_COCO, output=output, batch_size=batch_size, workers=workers)
        result = json.loads((output/f'{family}.json').read_text())
        assert result['images'] == total
        rows.append({'family':family, 'artifact':result['artifact'], 'resolution':result['resolution'],
            'images':total, 'seconds':result['seconds'], 'subset_map':float(row['map']),
            'difference_pp':100*(result['scores']['map']-float(row['map'])), **result['scores']})
        pd.DataFrame(rows).to_csv(output/'summary.csv', index=False)
        state['models'][family] = 'complete'
        write(output/'status.json', state)
    table = pd.DataFrame(rows).set_index('family')
    for name, columns in [('per_class',['Vehicle','VulnerableVehicle','Pedestrian']),
                          ('by_size',['map_small_reference1344','map_medium_reference1344','map_large_reference1344'])]:
        values = table[columns].replace(-1,np.nan)
        if name == 'by_size':
            values = values.rename(columns=dict(zip(columns,['Small','Medium','Large'])))
        ax = values.plot.bar(figsize=(11,5))
        ax.set(xlabel='',ylabel='Validation AP50–95', title=f'Full validation: {total:,} images'+
               ('; size bins at reference width 1344' if name=='by_size' else ''))
        ax.figure.tight_layout();ax.figure.savefig(output/f'{name}.png',dpi=150);plt.close(ax.figure)
    state.update(phase='complete', current_model=None)
    write(output/'status.json', state)
    print(table[['resolution','map','map_50','subset_map','difference_pp']].to_string(), flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['audit','sweep','pilots','summary','full-validation','all'],default='all')
    p.add_argument('--batch-size',type=int,default=4,help='Full-validation inference batch')
    p.add_argument('--workers',type=int,default=4,help='Full-validation image loader workers')
    args=p.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    OUTPUT.mkdir(parents=True,exist_ok=True);LOGS.mkdir(parents=True,exist_ok=True)
    if args.phase == 'full-validation':
        full_validation(args.batch_size,args.workers)
        return
    prepare()
    if args.phase in ['audit','all']:
        for family in MODELS:
            resolution=560 if family=='rf-detr' else 800 if MODELS[family].backend=='torchvision' else 512
            write(LOGS/'status.json',{'phase':'audit','family':family,'resolution':resolution})
            audit(family,ROOT/f'models/tuned/{family}',resolution,f'{family}-r{resolution}-frozen')
        for family,resolution in [('yolo',768),('rf-detr',840)]:
            write(LOGS/'status.json',{'phase':'audit','family':family,'resolution':resolution})
            audit(family,ROOT/f'models/tuned/{family}',resolution,f'{family}-r{resolution}-frozen')
    if args.phase in ['sweep','all']:
        for family,resolution in [('yolo',1024),('yolo',1280),('rf-detr',672),
                                  ('d-fine',768),('rt-detr',768),('detr',768)]:
            write(LOGS/'status.json',{'phase':'sweep','family':family,'resolution':resolution})
            audit(family,ROOT/f'models/tuned/{family}',resolution,f'{family}-r{resolution}-frozen')
    if args.phase in ['pilots','all']:
        pilots()
    summarize()
    write(LOGS/'status.json',{'phase':'complete','full_retraining_started':False})


if __name__=='__main__':
    main()
