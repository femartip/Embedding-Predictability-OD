"""Cache global, mean-patch and spatial image representations."""
import argparse
import time
import gc
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from .data import DEFAULT_DATA, Images, collate_samples, write_json
from .encoders import ENCODERS, Encoder


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=DEFAULT_DATA)
    p.add_argument('--encoders', nargs='+', choices=ENCODERS, default=ENCODERS)
    p.add_argument('--size', nargs=2, type=int, default=[224, 448], metavar=('HEIGHT', 'WIDTH'))
    p.add_argument('--grid', nargs=2, type=int, default=[2, 4])
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--device', default='cuda')
    p.add_argument('--revision', default='main')
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(4)
    rows = pd.read_csv(args.data / 'images.csv')
    base = args.data / f'features/{args.size[0]}x{args.size[1]}-grid{args.grid[0]}x{args.grid[1]}'
    base.mkdir(parents=True, exist_ok=True)
    def loader():
        return DataLoader(Images(rows, tensors=True), batch_size=args.batch_size,
            num_workers=args.workers, collate_fn=collate_samples, prefetch_factor=1 if args.workers else None)
    def images(tensors):
        return [Image.fromarray(t.permute(1, 2, 0).numpy()) for t in tensors]
    if not (base / 'quality.npz').exists():
        quality = []
        for tensors, _ in tqdm(loader(), desc='image quality'):
            for image in images(tensors):
                image.thumbnail((448, 224))
                gray = np.asarray(image.convert('L'), dtype=np.float32)/255
                quality.append([gray.mean(), gray.std(), cv2.Laplacian(gray, cv2.CV_32F).var()])
        np.savez_compressed(base / 'quality.npz', image_id=rows.image_id.to_numpy(), features=np.array(quality, dtype=np.float32))
    for name in args.encoders:
        if all((base / f'{name}-{key}.json').exists() for key in ['global', 'mean', 'spatial']):
            print('Completed:', name, flush=True)
            continue
        started = time.perf_counter()
        encoder = Encoder(name, args.size, args.grid, args.revision).to(args.device).eval()
        arrays = {key: [] for key in ['global', 'mean', 'spatial']}
        with torch.no_grad():
            for tensors, _ in tqdm(loader(), desc=name):
                inputs = encoder.preprocess(images(tensors))
                features = encoder({k: v.to(args.device) for k, v in inputs.items()})
                assert all(torch.isfinite(v).all() for v in features.values())
                for key, value in features.items():
                    arrays[key].append(value.cpu().numpy())
        for key, chunks in arrays.items():
            path = base / f'{name}-{key}.npz'
            np.savez_compressed(path, image_id=rows.image_id.to_numpy(), features=np.concatenate(chunks))
            write_json(path.with_suffix('.json'), encoder.config() | {'pool': key,
                'dimension': chunks[0].shape[1], 'parameters': sum(p.numel() for p in encoder.parameters()),
                'seconds': time.perf_counter()-started})
        del encoder, features, arrays, chunks
        gc.collect()
        torch.cuda.empty_cache()
    print(base)


if __name__ == '__main__':
    main()
