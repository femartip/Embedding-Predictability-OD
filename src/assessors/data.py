import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.dataset import ROOT

DEFAULT_DATA = ROOT / 'data/assessors'


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))


def read_data(folder):
    return pd.read_csv(Path(folder) / 'images.csv'), pd.read_csv(Path(folder) / 'targets.csv')


class Images(Dataset):
    def __init__(self, rows):
        self.rows = rows.reset_index(drop=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        return Image.open(row.filepath).convert('RGB'), int(row.image_id)


class ImageBatch:
    def __init__(self, encoder, rows, columns=()):
        self.encoder = encoder
        self.targets = rows.set_index('image_id')[list(columns)]

    def __call__(self, batch):
        images, ids = zip(*batch)
        x = self.encoder.preprocess(images)
        y = torch.tensor(self.targets.loc[list(ids)].to_numpy(dtype=np.float32))
        return x, y
