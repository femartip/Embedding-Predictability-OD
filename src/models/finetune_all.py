"""Run the detector panel sequentially on one GPU, logging each training run."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import shlex
from datetime import datetime, timezone

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.catalog import MODELS, resolve_model
from src.data.dataset import ROOT


def training_command(args, family, recipes):
    command = [sys.executable, '-u', str(ROOT / 'src/models/train.py'), family,
               '--device', 'cuda', '--output-dir', str(args.artifacts_dir/family), '--coco-dir', str(args.coco_dir)]
    recipe = {k: v for k, v in recipes.get(family, {}).items() if k not in ('profile', 'pilot')}
    recipe['workers'] = args.workers if args.workers is not None else recipe.get('workers', 4)
    for key, value in recipe.items():
        flag = key.replace('_', '-')
        command += [f'--{flag}' if value else f'--no-{flag}'] if isinstance(value, bool) else [f'--{flag}', str(value)]
    if MODELS[family].backend == 'torchvision' and family not in recipes:
        command += ['--batch-size', '2', '--grad-accum-steps', '8']
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', default=list(MODELS))
    parser.add_argument('--workers', type=int)
    parser.add_argument('--log-dir', type=Path, default=ROOT / 'logs/finetune')
    parser.add_argument('--artifacts-dir', type=Path, default=ROOT / 'models')
    parser.add_argument('--recipes', type=Path, help='Settings selected by tune.py')
    parser.add_argument('--coco-dir', type=Path, default=ROOT / 'data/zod_coco')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    families = [resolve_model(name)[0] for name in args.models]
    recipes = json.loads(args.recipes.read_text()) if args.recipes else {}
    if args.dry_run:
        for family in families:
            print(shlex.join(training_command(args, family, recipes)))
        return
    args.log_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.log_dir / 'status.json'
    state = {'pid': os.getpid(), 'started_at': datetime.now(timezone.utc).isoformat(),
             'models': {family: {'status': 'pending'} for family in families}}

    def save():
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(status_path)

    save()
    for family in families:
        if (args.artifacts_dir / family / 'manifest.json').exists():
            state['models'][family] = {'status': 'already_complete'}
            save()
            continue
        command = training_command(args, family, recipes)
        log_path = args.log_dir / f'{family}.log'
        with log_path.open('a', buffering=1) as log:
            started_at = datetime.now(timezone.utc).isoformat()
            log.write(f'\n{started_at}: {command}\n')
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, 'WANDB_MODE': 'offline', 'TOKENIZERS_PARALLELISM': 'false'})
            state['models'][family] = {'status': 'running', 'pid': process.pid,
                'started_at': started_at, 'log': str(log_path), 'command': command}
            save()
            result = process.wait()
        state['models'][family].update(status='complete' if result == 0 else 'failed',
            exit_code=result, finished_at=datetime.now(timezone.utc).isoformat())
        save()
        if result:
            raise SystemExit(f'{family} failed; see {log_path}. Remaining models have not started.')
    state['finished_at'] = datetime.now(timezone.utc).isoformat()
    save()


if __name__ == '__main__':
    main()
