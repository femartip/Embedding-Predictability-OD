# Code from:
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
import os
import contextlib
import json
from collections import defaultdict
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from pathlib import Path
import shutil
from tqdm import tqdm
import yaml

def make_dirs(dir):
    """Creates a directory with subdirectories 'labels' and 'images', removing existing ones."""
    dir = Path(dir)
    if dir.exists():
        shutil.rmtree(dir)  # delete dir
    for p in dir, dir / "labels", dir / "images":
        p.mkdir(parents=True, exist_ok=True)  # make dir
    return dir

def convert_coco_json(json_dir: str, save_dir:str, copy_images: bool=True) -> None:
    """Converts COCO JSON format to YOLO label format, with options for segments and class mapping."""
    save_dir = make_dirs(save_dir)  # type: ignore # output directory 
    # Import json
    for json_file in sorted(Path(json_dir).resolve().glob("*.json")):
        split = os.path.basename(json_file).split("_")[-1].strip(".json") 
        fn = Path(save_dir) / "labels" / split  # folder name
        fn.mkdir()
        print(f"Reading {json_file}")
        with open(json_file) as f:
            data = json.load(f)

        # Create image dict
        categories_dict = {item['id'] - 1: item['name'] for item in data['categories']}
        images = {"{:g}".format(x["id"]): x for x in data["images"]}
        print(f"Number of images found {len(images)}")
        # Create image-annotations dict
        imgToAnns = defaultdict(list)
        for ann in data["annotations"]:
            imgToAnns[ann["image_id"]].append(ann)

        # Write labels file
        for img_id, anns in tqdm(imgToAnns.items(), desc=f"Annotations {json_file}"):
            #print(f"Loading image {img_id}")
            img = images[f"{img_id:g}"]
            h, w, f = img["height"], img["width"], img["file_name"]
            file_name = f.split("/")[-1]

            bboxes = []
            for ann in anns:
                #if ann["iscrowd"]:
                #    print(f"Not saving ann from {img_id} as crowd")
                #    continue
                # The COCO box format is [top left x, top left y, width, height]
                box = np.array(ann["bbox"], dtype=np.float64)
                x, y, bw, bh = box.tolist()
                x1, y1 = x, y
                x2, y2 = x + bw, y + bh

                x1 = np.clip(x1, 0, w); y1 = np.clip(y1, 0, h); x2 = np.clip(x2, 0, w); y2 = np.clip(y2, 0, h)

                bw = x2 - x1
                bh = y2 - y1
                if bw <= 0 or bh <= 0:
                    continue

                xc = (x1 + x2) / 2
                yc = (y1 + y2) / 2
                box = np.array([xc, yc, bw, bh], dtype=np.float64)

                box[[0, 2]] /= w  # normalize x and width by image width
                box[[1, 3]] /= h  # normalize y and height by image height
                
                if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                    continue

                cls = ann["category_id"] - 1  # class
                box = [cls] + box.tolist()
                if box not in bboxes:
                    bboxes.append(box)
            
            if copy_images:
                img = cv2.imread(f)  # Read image
                if img is not None:
                    image_folder = Path(os.path.join(save_dir, "images", split))
                    image_folder.mkdir(exist_ok=True)
                    save_path = Path(os.path.join(image_folder, file_name))
                    cv2.imwrite(save_path, img)  # type: ignore # Save image

            # Write
            with open((fn / file_name).with_suffix(".txt"), "a") as file:
                for i in range(len(bboxes)):
                    line = (*(bboxes[i]),)  # cls, box or segments
                    file.write(("%g " * len(line)).rstrip() % line + "\n")

    yaml_dict = {
        # "path": "yolo_datasets",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {k: v for k, v in categories_dict.items()}
    }

    with open((Path(save_dir) / "dataset").with_suffix('.yaml'), "w") as f:
        yaml.dump(yaml_dict, f)
        
            
if __name__ == "__main__":
    convert_coco_json("./data/zod_coco/", "./data/zod_yolo/", copy_images=True)