"""This module will generate a COCO JSON file from the ZOD dataset."""

import json
import os
import random
from functools import partial
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import numpy as np

from tqdm.contrib.concurrent import process_map

from zod import ZodFrames
from zod.anno.object import OBJECT_CLASSES, ObjectAnnotation
from zod.cli.utils import Version
from zod.constants import AnnotationProject, Anonymization
from zod.data_classes.frame import ZodFrame
from zod.utils.utils import str_from_datetime

# Map classes to categories, starting from 1
CATEGORY_NAME_TO_ID = {cls: i + 1 for i, cls in enumerate(OBJECT_CLASSES)}
OPEN_DATASET_URL = "https://www.ai.se/en/data-factory/datasets/data-factory-datasets/zenseact-open-dataset"

VAL_TILE_FRACTION = 0.2
TILE_SIZE_DEG = 0.01  # approx 1km
RANDOM_SEED = 43


#Val split
def tile_id(lat, lon, size_deg):
    lat_bin = np.floor(lat / size_deg) * size_deg
    lon_bin = np.floor(lon / size_deg) * size_deg
    return f'{lat_bin:.4f}_{lon_bin:.4f}'



def _convert_frame(frame: ZodFrame, classes: List[str], anonymization: Anonymization, use_png: bool) -> Tuple[dict, List[dict]]:
    objs: List[ObjectAnnotation] = frame.get_annotation(AnnotationProject.OBJECT_DETECTION)
    camera_frame = frame.info.get_key_camera_frame(anonymization=anonymization)
    file_name = camera_frame.filepath

    if anonymization == Anonymization.ORIGINAL:
        file_name = file_name.replace(Anonymization.BLUR.value, Anonymization.ORIGINAL.value)
    if use_png:
        file_name = file_name.replace(".jpg", ".png")

    image_dict = {
        "id": int(frame.info.id),
        "license": 1,
        "file_name": file_name,
        "height": camera_frame.height,
        "width": camera_frame.width,
        "date_captured": str_from_datetime(frame.info.keyframe_time),
    }
    anno_dicts = [
        {
            # avoid collisions by assuming max 1k objects per frame
            "id": int(frame.info.id) * 1000 + obj_idx,
            "image_id": int(frame.info.id),
            "category_id": CATEGORY_NAME_TO_ID[obj.name],
            "bbox": [round(val, 2) for val in obj.box2d.xywh.tolist()],
            "area": round(obj.box2d.area, 2),
            "iscrowd": obj.subclass == "Unclear",
        }
        for obj_idx, obj in enumerate(objs) if obj.name in classes and str(obj.occlusion_level) in ["None", "Light", "Medium"] and (obj.box2d.ymax - obj.box2d.ymin) >= 25
    ]
    return image_dict, anno_dicts


def generate_coco_json(dataset: ZodFrames,split: str,classes: List[str],anonymization: Anonymization,use_png: bool,frame_ids: List[str] | None = None,desc: str | None = None,) -> dict:
    """Generate COCO JSON file from the ZOD dataset."""
    if frame_ids is None:
        assert split in ["train", "val"], f"Unknown split: {split}"
        frame_ids = list(dataset.get_split(split))      #type: ignore
    frame_infos = [dataset[frame_id] for frame_id in frame_ids]
    _convert_frame_w_classes = partial(_convert_frame, classes=classes, anonymization=anonymization, use_png=use_png)
    results = process_map(
        _convert_frame_w_classes,
        frame_infos,
        desc=desc or f"Converting {split} frames",
        chunksize=50 if dataset._version == "full" else 1,
    )
    results = [(img, annos) for img, annos in results if len(annos) > 0]
    image_dicts, all_annos = zip(*results)
    anno_dicts = [anno for annos in all_annos for anno in annos]  # flatten
    coco_json = {
        "images": image_dicts,
        "annotations": anno_dicts,
        "info": {
            "description": "Zenseact Open Dataset",
            "url": OPEN_DATASET_URL,
            "version": dataset._version,  # TODO: add dataset versioning
            "year": 2022,
            "contributor": "ZOD team",
            "date_created": "2022/12/15",
        },
        "licenses": [
            {
                "url": "https://creativecommons.org/licenses/by-sa/4.0/",
                "name": "Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)",
                "id": 1,
            },
        ],
        "categories": [
            {"supercategory": "object", "id": category_id, "name": category_name}
            for category_name, category_id in CATEGORY_NAME_TO_ID.items()
            if category_name in classes
        ],
    }
    return coco_json


# Use typer instead of argparse
def convert_to_coco(dataset_root: str, output_dir: str, version: str = "full", anonymization: Anonymization = Anonymization.BLUR, use_png: bool = False):
    #classes = OBJECT_CLASSES
    classes = ['Vehicle', 'Pedestrian', 'VulnerableVehicle']
    # check that classes are valid
    for cls in classes:
        if cls not in OBJECT_CLASSES:
            raise ValueError(f"ERROR: Invalid class: {cls}.")
    print(f"Converting ZOD to COCO format. Version: {version}, anonymization: {anonymization}, classes: {classes}, filtering by: occlusion level either None, Light or Medium and height >= 25")
    
    zod_frames = ZodFrames(str(dataset_root), version)      #type: ignore

    base_name = f"zod_{version}_{anonymization}"
    if use_png:
        base_name += "_png"

    os.makedirs(output_dir, exist_ok=True)

    ## Make split of train into train and val by tiles
    train_ids = list(zod_frames.get_split("train"))
    coords = {frame_id: {"lat": zod_frames[frame_id].metadata.latitude, "lon": zod_frames[frame_id].metadata.longitude} for frame_id in train_ids}
    df = pd.DataFrame.from_dict(coords, orient='index')
    df['tile_id'] = [tile_id(lat, lon, TILE_SIZE_DEG) for lat, lon in zip(df['lat'], df['lon'])]
    
    unique_tiles = sorted(set(df['tile_id'].dropna().unique()))
    rng = np.random.default_rng(RANDOM_SEED)
    val_tile_count = int(len(unique_tiles) * VAL_TILE_FRACTION)
    val_tiles = set(rng.choice(unique_tiles, size=val_tile_count, replace=False))
    df['split'] = np.where(df['tile_id'].isin(val_tiles), 'val', 'train')
    val_ids = df[df['split'] == 'val'].index.tolist()
    train_ids = df[df['split'] == 'train'].index.tolist()
    print(f"Total train frames: {len(train_ids)}, val frames: {len(val_ids)}")

    coco_json_train = generate_coco_json(
        zod_frames,
        split="train",
        classes=classes,
        anonymization=anonymization,
        use_png=use_png,
        frame_ids=train_ids,
        desc="Converting train frames (80% of original train)",
    )
    with open(os.path.join(output_dir, f"{base_name}_train.json"), "w") as f:
        json.dump(coco_json_train, f)

    coco_json_val = generate_coco_json(
        zod_frames,
        split="train",
        classes=classes,
        anonymization=anonymization,
        use_png=use_png,
        frame_ids=val_ids,
        desc="Converting val (20% of original train) frames",
    )
    with open(os.path.join(output_dir, f"{base_name}_val.json"), "w") as f:
        json.dump(coco_json_val, f)

    coco_json_test = generate_coco_json(
        zod_frames,
        split="val",
        classes=classes,
        anonymization=anonymization,
        use_png=use_png,
        desc="Converting test (original val) frames",
    )
    with open(os.path.join(output_dir, f"{base_name}_test.json"), "w") as f:
        json.dump(coco_json_test, f)

    print("Successfully converted ZOD to COCO format. Output files:")
    print(f"    train:  {output_dir}/{base_name}_train.json")
    print(f"    val:    {output_dir}/{base_name}_val.json")
    print(f"    test:   {output_dir}/{base_name}_test.json")


if __name__ == "__main__":
    dataset_root = "./data/zod"
    output_dir = "./data/zod_coco"
    str_version = "full"
    convert_to_coco(dataset_root=dataset_root, output_dir=output_dir, version=str_version)  
