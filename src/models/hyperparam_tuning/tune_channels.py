"""Keep channels-last layout only when a probe and short learning pilot support it."""
import sys
from pathlib import Path
import argparse
import json
import math
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.models.catalog import MODELS
from src.data.dataset import ROOT
from src.models.hyperparam_tuning.tune import execute, save, histories


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',nargs='+',default=list(MODELS))
    args = p.parse_args()
    path = ROOT/'data/tuning/recipes.json'
    recipes = json.loads(path.read_text())
    coco, logs = ROOT/'data/tuning/coco', ROOT/'logs/tuning'
    for family in args.models:
        recipe = recipes[family]
        probe = logs/f'{family}-channels.json'
        if not probe.exists():
            code = execute(['src/models/hyperparam_tuning/profile_training.py',family,'--coco-dir',coco,
                '--batch-size',recipe['batch_size'],'--precision',recipe['precision'] if recipe['amp'] else 'fp32',
                '--rf-groups',recipe.get('rf_groups',13),'--channels-last','--output',probe],logs/f'{family}-channels.log')
            if code:
                continue
        result = json.loads(probe.read_text())
        if result['images_per_second'] < 1.05*recipe['profile']['images_per_second'] or result['peak_gib']>20:
            print(family,'channels-last not faster enough',flush=True)
            continue
        parent = ROOT/f'models/tuning/{family}-channels'
        trial = parent/family
        done = trial/'pilot.json'
        if done.exists():
            pilot = json.loads(done.read_text())
        else:
            epochs=int(recipe['pilot']['epochs'][-1])
            selected = {k:v for k,v in recipe.items() if k not in ['profile','pilot','epochs','max_steps','eval_every']}
            selected.update(channels_last=True,patience=epochs+1,warmup_ratio=0. if family=='yolo' else .1)
            command = ['src/models/train.py',family,'--coco-dir',coco,'--output-dir',trial]
            for key,value in selected.items():
                flag=key.replace('_','-')
                command += [f'--{flag}' if value else f'--no-{flag}'] if isinstance(value,bool) else [f'--{flag}',value]
            if MODELS[family].backend=='torchvision':
                steps=math.ceil(768/(recipe['batch_size']*recipe['grad_accum_steps']))
                command += ['--max-steps',steps*epochs,'--eval-every',steps]
            else:
                command += ['--epochs',epochs]
            started=time.perf_counter()
            if execute(command,logs/f'{family}-channels-pilot.log'):
                continue
            history=histories(family,parent)
            val=[r for r in history if r.get('val_map') is not None]
            pilot={'family':family,'lr':recipe['lr'],'channels_last':True,'seconds':time.perf_counter()-started,
                'semantic_init':recipe.get('semantic_init',False),
                'validation_maps':[r['val_map'] for r in val],'epochs':[r['epoch'] for r in val],
                'history':history,'best_map':max(r['val_map'] for r in val),
                'learning_verified':max(r['val_map'] for r in val[1:])>val[0]['val_map']}
            if family=='rf-detr':
                pilot['rf_groups']=recipe.get('rf_groups',13)
            save(done,pilot)
        if pilot['learning_verified'] and pilot['best_map']>=recipe['pilot']['best_map']-.02:
            recipe.update(channels_last=True,profile=result,pilot=pilot)
            save(path,recipes)
            print(family,'channels-last selected',flush=True)


if __name__=='__main__':
    main()
