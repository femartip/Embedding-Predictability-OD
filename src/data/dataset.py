"""Reuse the existing COCO splits without moving or regenerating ZOD data."""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = ROOT.parent / "Validators-OD"
DEFAULT_COCO = ROOT / "data/zod_coco"


def annotation_file(coco_dir, split):
    files = sorted(Path(coco_dir).glob(f"*_{split}.json"))
    if len(files) != 1:
        raise ValueError(f"Expected one *_{split}.json in {coco_dir}; found {len(files)}")
    return files[0]


def resolve_image(filename, image_root=None):
    path = Path(filename)
    if image_root is not None:
        root = Path(image_root).resolve()
        # --image-root points at zod/, rather than at the repository.
        parts = path.parts
        relative = Path(*parts[parts.index("single_frames"):]) if "single_frames" in parts else Path(path.name)
        candidates = [root / relative]
    else:
        candidates = [path, ROOT / path, REFERENCE_ROOT / path]
        if "single_frames" in path.parts:
            relative = Path(*path.parts[path.parts.index("single_frames"):])
            candidates += [ROOT / "data/zod" / relative, REFERENCE_ROOT / "data/zod" / relative]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Missing image {filename}; pass --image-root /path/to/zod")


def load_coco(coco_dir, split, image_root=None, max_samples=None):
    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be positive")
    path = annotation_file(coco_dir, split)
    data = json.loads(path.read_text())
    categories = sorted(data["categories"], key=lambda c: c["id"])
    ids = [c["id"] for c in categories]
    classes = [c["name"] for c in categories]
    if len(ids) != len(set(ids)) or len(classes) != len(set(classes)):
        raise ValueError(f"Duplicate categories in {path}")
    annotations = defaultdict(list)
    for ann in data["annotations"]:
        if ann["category_id"] not in ids:
            raise ValueError(f"Unknown category in annotation {ann['id']}")
        annotations[ann["image_id"]].append(ann)
    images = data["images"][:max_samples]
    if len({im['id'] for im in images}) != len(images):
        raise ValueError(f"Duplicate image IDs in {path}")
    for image in images:
        image["filepath"] = str(resolve_image(image["file_name"], image_root))
    return {"images": images, "annotations": annotations, "categories": categories,
            "classes": classes, "category_to_index": {cid: i for i, cid in enumerate(ids)}}


def validate_splits(coco_dir):
    seen, categories = set(), None
    for split in ("train", "val", "test"):
        data = json.loads(annotation_file(coco_dir, split).read_text())
        current = sorted(data["categories"], key=lambda c: c["id"])
        if categories is not None and current != categories:
            raise ValueError(f"Inconsistent categories in {split}")
        categories = current
        ids = {im["id"] for im in data["images"]}
        if seen & ids:
            raise ValueError(f"Overlapping image IDs in {split}")
        seen |= ids


def clipped_annotations(image, annotations):
    """Clip COCO xywh boxes at the image boundary, as the old YOLO exporter did."""
    result = []
    for ann in annotations:
        x, y, w, h = ann["bbox"]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(image["width"], x + w), min(image["height"], y + h)
        if x2 > x1 and y2 > y1:
            result.append({**ann, "bbox": [x1, y1, x2-x1, y2-y1], "area": (x2-x1)*(y2-y1)})
    return result


def as_fiftyone(data):
    import fiftyone as fo
    dataset = fo.Dataset()
    dataset.default_classes = data["classes"]
    names = {c["id"]: c["name"] for c in data["categories"]}
    def samples():
        for im in data["images"]:
            detections = []
            for ann in clipped_annotations(im, data["annotations"][im["id"]]):
                x, y, w, h = ann["bbox"]
                detections.append(fo.Detection(label=names[ann["category_id"]],
                    bounding_box=[x/im['width'], y/im['height'], w/im['width'], h/im['height']],
                    iscrowd=bool(ann.get('iscrowd', 0))))
            yield fo.Sample(filepath=im['filepath'], image_id=im['id'],
                metadata=fo.ImageMetadata(width=im['width'], height=im['height']),
                ground_truth=fo.Detections(detections=detections))
    try:
        dataset.add_samples(samples())
    except Exception:
        dataset.delete()
        raise
    return dataset


def prepare_native_data(coco_dir, image_root, output_dir, backend):
    """Generate backend labels and symlink source images; never delete source data."""
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        data = load_coco(coco_dir, split, image_root)
        folder = output / ({"val": "valid"}.get(split, split) if backend == "rfdetr" else f"images/{split}")
        folder.mkdir(parents=True, exist_ok=True)
        images, annotations = [], []
        for im in data['images']:
            name = f"{im['id']:06d}{Path(im['filepath']).suffix}"
            link = folder / name
            target = Path(im['filepath'])
            if link.is_symlink() and link.resolve() == target:
                pass
            elif link.exists() or link.is_symlink():
                raise FileExistsError(f"Refusing to replace {link}")
            else:
                link.symlink_to(target)
            images.append({k: v for k, v in im.items() if k != 'filepath'} | {'file_name': name})
            anns = clipped_annotations(im, data['annotations'][im['id']])
            annotations.extend(anns)
            if backend == 'ultralytics':
                labels = output / 'labels' / split
                labels.mkdir(parents=True, exist_ok=True)
                lines = []
                for ann in anns:
                    x, y, w, h = ann['bbox']
                    cls = data['category_to_index'][ann['category_id']]
                    lines.append(f"{cls} {(x+w/2)/im['width']} {(y+h/2)/im['height']} {w/im['width']} {h/im['height']}")
                (labels / Path(name).with_suffix('.txt')).write_text('\n'.join(lines))
        if backend == 'rfdetr':
            # RF-DETR's Roboflow loader requires contiguous category IDs starting at 0.
            categories = [{'id': i, 'name': n, 'supercategory': 'object'} for i, n in enumerate(data['classes'])]
            annotations = [{**a, 'category_id': data['category_to_index'][a['category_id']]} for a in annotations]
            (folder / '_annotations.coco.json').write_text(json.dumps({'images': images, 'annotations': annotations, 'categories': categories}))
    if backend == 'ultralytics':
        # JSON is valid YAML, including names as a list.
        (output / 'dataset.yaml').write_text(json.dumps({'path': str(output), 'train': 'images/train',
            'val': 'images/val', 'test': 'images/test', 'names': data['classes']}, indent=2))
    return output
