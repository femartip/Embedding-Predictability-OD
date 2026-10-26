"""Compare fewer RF-DETR training query groups after the learning-rate pilots."""
import argparse
import sys
import json
from pathlib import Path
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data.dataset import ROOT
from src.models.hyperparam_tuning.tune import execute, save, histories


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--groups', nargs='+', type=int, default=[1,3])
    p.add_argument('--epochs', type=int, default=3)
    args = p.parse_args()
    logs, coco = ROOT/'logs/tuning', ROOT/'data/tuning/coco'
    path = ROOT/'data/tuning/recipes.json'
    recipes = json.loads(path.read_text())
    original = recipes['rf-detr']
    candidates = [original]
    for groups in args.groups:
        probes = []
        for batch in [4,8,16]:
            tag = f'rf-detr-g{groups}-bf16-b{batch}'
            result_path = logs/f'{tag}.json'
            if not result_path.exists():
                code = execute(['src/models/hyperparam_tuning/profile_training.py','rf-detr','--coco-dir',coco,
                    '--batch-size',batch,'--precision','bf16','--rf-groups',groups,'--output',result_path],logs/f'{tag}.log')
                if code:
                    break
            result = json.loads(result_path.read_text())
            probes.append(result)
            print(tag,result['images_per_second'],result['peak_gib'],flush=True)
        best = max([r for r in probes if r['peak_gib']<20],key=lambda r:r['images_per_second'])
        batch = best['batch_size']
        parent = ROOT/f'models/tuning/rf-detr-g{groups}-lr{original["lr"]:g}'
        trial = parent/'rf-detr'
        done = trial/'pilot.json'
        if done.exists():
            pilot = json.loads(done.read_text())
        else:
            started = time.perf_counter()
            code = execute(['src/models/train.py','rf-detr','--coco-dir',coco,'--output-dir',trial,
                '--epochs',args.epochs,'--batch-size',batch,'--grad-accum-steps',max(1,16//batch),
                '--workers',4,'--lr',original['lr'],'--backbone-lr',1e-5,'--rf-groups',groups,
                '--warmup-ratio','.1','--patience',args.epochs+1],logs/f'rf-detr-g{groups}-pilot.log')
            assert code == 0, f'RF-DETR group {groups} pilot failed'
            history = histories('rf-detr',parent)
            val = [r for r in history if r.get('val_map') is not None]
            pilot = {'family':'rf-detr','lr':original['lr'],'rf_groups':groups,'seconds':time.perf_counter()-started,
                'validation_maps':[r['val_map'] for r in val],'epochs':[r['epoch'] for r in val],
                'history':history,'best_map':max(r['val_map'] for r in val),
                'learning_verified':max(r['val_map'] for r in val[1:])>val[0]['val_map']}
            save(done,pilot)
        candidates.append({**original,'rf_groups':groups,'batch_size':batch,'grad_accum_steps':max(1,16//batch),
                           'profile':best,'pilot':pilot})
        print('Groups',groups,'AP',pilot['validation_maps'],flush=True)
    reference = max(r['pilot']['best_map'] for r in candidates)
    eligible = [r for r in candidates if r['pilot']['learning_verified'] and r['pilot']['best_map']>=reference-.02]
    selected = max(eligible,key=lambda r:r['profile']['images_per_second'])
    recipes['rf-detr'] = selected
    save(path,recipes)
    print('Selected groups',selected.get('rf_groups',13),'based on speed with AP within 0.02 of best short pilot.',flush=True)


if __name__ == '__main__':
    main()
