"""Fine-tune Zoo detectors with backend-specific ZOD recipes."""
import argparse
import sys
import json
import math
from datetime import datetime, timezone
import random
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.catalog import MODELS, resolve_model
from src.data.dataset import (DEFAULT_COCO, ROOT, load_coco, clipped_annotations,
                              prepare_native_data, validate_splits)
from src.models.adapters import load_pretrained, replace_torchvision_head, save_manifest


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model', help='Family alias or full Zoo name; see run_inference.py --list-models')
    p.add_argument('--coco-dir', type=Path, default=DEFAULT_COCO)
    p.add_argument('--image-root', type=Path, help='Root containing single_frames; auto-detects Validators-OD')
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--resume', type=Path, help='Backend checkpoint (optimizer state required for full resumption)')
    p.add_argument('--epochs', type=int)
    p.add_argument('--batch-size', type=int)
    p.add_argument('--grad-accum-steps', type=int, help='Accumulate microbatches before each optimizer step')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--max-steps', type=int, help='Override iteration budget; useful for a smoke run')
    p.add_argument('--seed', type=int, default=43)
    p.add_argument('--lr', type=float)
    p.add_argument('--backbone-lr', type=float, default=1e-5)
    p.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--precision', choices=['bf16', 'fp16'], default='bf16')
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--eval-every', type=int, default=5000)
    p.add_argument('--warmup-ratio', type=float, default=.05)
    p.add_argument('--optimizer', default='auto')
    p.add_argument('--init-checkpoint', type=Path, help='Weights only; starts a new optimization schedule')
    p.add_argument('--deterministic', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--rf-groups', type=int, default=13)
    p.add_argument('--channels-last', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--semantic-init', action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    family, spec = resolve_model(args.model)
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError('--max-steps must be positive')
    if args.epochs is not None and args.epochs < 1:
        raise ValueError('--epochs must be positive')
    if args.grad_accum_steps is not None and args.grad_accum_steps < 1:
        raise ValueError('--grad-accum-steps must be positive')
    if args.grad_accum_steps and spec.backend == 'ultralytics':
        raise ValueError('YOLO manages gradient accumulation internally')
    if args.workers < 0:
        raise ValueError('--workers must be nonnegative')
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError('--batch-size must be positive')
    if args.max_steps and spec.backend in ('ultralytics', 'rfdetr'):
        raise ValueError('Use --epochs for native YOLO/RF-DETR smoke runs')
    if args.epochs and spec.backend == 'torchvision':
        raise ValueError('Torchvision uses --max-steps, matching the original iteration schedule')
    import numpy as np
    import torch
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable to this process. Run training with GPU device access enabled.')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = not args.deterministic
    validate_splits(args.coco_dir)
    output = (args.output_dir or ROOT / 'models' / family).resolve()
    if (output / 'manifest.json').exists() and not args.resume:
        raise FileExistsError(f'{output} already contains a trained model; use --resume or a new --output-dir')
    output.mkdir(parents=True, exist_ok=True)
    # Read the real category list rather than the full ZOD ontology.
    categories = json.loads(next(args.coco_dir.glob('*_train.json')).read_text())['categories']
    classes = [c['name'] for c in sorted(categories, key=lambda c: c['id'])]
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    if spec.backend in ('ultralytics', 'rfdetr'):
        native_data = prepare_native_data(args.coco_dir, args.image_root, output / 'dataset', spec.backend)
        wrapper = load_pretrained(family, args.device)
        model = wrapper._model  # Backend models exposed by the pinned FiftyOne wrappers.
        if spec.backend == 'ultralytics':
            from ultralytics import YOLO
            from ultralytics.models.yolo.detect import DetectionTrainer
            # The inference wrapper sets up a fused predictor. Reload the original
            # checkpoint so fine-tuning retains unfused layers and trainable weights.
            model = YOLO(str(args.resume or args.init_checkpoint or model.ckpt_path))
            class FastDetectionTrainer(DetectionTrainer):
                def preprocess_batch(self, batch):
                    batch = super().preprocess_batch(batch)
                    batch['img'] = batch['img'].contiguous(memory_format=torch.channels_last)
                    return batch
            recipe = dict(data=str(native_data / 'dataset.yaml'), project=str(output), name='train',
                workers=args.workers, amp=args.amp, save=True, save_period=10, imgsz=512, rect=True,
                pretrained=True, seed=args.seed, device=args.device, patience=args.patience,
                epochs=args.epochs or 50, batch=args.batch_size or 32, plots=True,
                exist_ok=True, lr0=args.lr or .001, optimizer=args.optimizer, deterministic=args.deterministic,
                warmup_epochs=args.warmup_ratio*(args.epochs or 50),
                multi_scale=0.25, resume=bool(args.resume))
            model.train(trainer=FastDetectionTrainer if args.channels_last else None, **recipe)
            checkpoint = output / 'train/weights/best.pt'
        else:
            from functools import partial
            import rfdetr.main as rf_main
            import rfdetr.engine as rf_engine
            rf_engine.CocoEvaluator = RFCocoEvaluator
            rf_main.DataLoader = partial(rf_main.DataLoader, pin_memory=True, persistent_workers=args.workers > 0)
            if args.rf_groups != 13:
                config = model.model_config.model_dump()
                config['group_detr'] = args.rf_groups
                model = type(model)(**config)
                del wrapper
            if args.channels_last:
                model.model.model.to(memory_format=torch.channels_last)
            recipe = dict(dataset_dir=str(native_data), epochs=args.epochs or 30,
                batch_size=args.batch_size or 4, grad_accum_steps=args.grad_accum_steps or 4, lr=args.lr or 1e-4,
                lr_encoder=args.backbone_lr, output_dir=str(output), warmup_epochs=args.warmup_ratio*(args.epochs or 30),
                early_stopping=True, early_stopping_patience=args.patience, tensorboard=True, amp=args.amp,
                num_workers=args.workers, multi_scale=False, expanded_scales=False,
                use_ema=False, run_test=False, resolution=560, seed=args.seed)
            recipe['group_detr'] = args.rf_groups
            if args.resume:
                recipe['resume'] = str(args.resume)
            def record_validation(stats):
                write_model_info(output, model.model.model, classes, recipe)
                rfdetr_previews(model, args.coco_dir, args.image_root, output, classes, int(stats['epoch']) + 1)
            model.callbacks['on_fit_epoch_end'].append(record_validation)
            model.train(**recipe)
            checkpoint = output / 'checkpoint_best_total.pth'
        settings['recipe'] = recipe
    elif spec.backend == 'torchvision':
        checkpoint, recipe = train_torchvision(family, args, output, classes)
        settings['recipe'] = recipe
    else:
        checkpoint, recipe = train_transformer(family, args, output, classes)
        settings['recipe'] = recipe
    save_manifest(output, family, classes, checkpoint, settings)
    print(f'Fine-tuned artifact: {output}')


def append_history(output, record):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    nonfinite = [key for key, value in record.items() if isinstance(value, float) and not math.isfinite(value)]
    record = {key: (None if key in nonfinite else value) for key, value in record.items()}
    if nonfinite:
        record['nonfinite_fields'] = nonfinite
    with (output / 'learning_history.jsonl').open('a') as handle:
        handle.write(json.dumps({'time': datetime.now(timezone.utc).isoformat(), **record}, allow_nan=False) + '\n')


def map_metrics(result, classes, label_offset=0):
    """Convert TorchMetrics results into scalar fields, including class AP."""
    values = {key: float(result[key]) for key in ['map', 'map_50', 'map_75', 'mar_100']}
    ids = result.get('classes')
    per_class = result.get('map_per_class')
    if ids is not None and per_class is not None:
        for label, score in zip(ids.reshape(-1).tolist(), per_class.reshape(-1).tolist()):
            index = int(label) - label_offset
            if 0 <= index < len(classes):
                values[f'map_{classes[index]}'] = float(score)
    return values


def save_previews(output, samples, classes, progress, confidence=0.3):
    """Render GT and predicted boxes beside each other, in source pixel space."""
    from PIL import Image, ImageDraw, ImageFont
    output = Path(output) / 'learning_previews'
    output.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=14)
    metadata = []
    for index, sample in enumerate(samples):
        with Image.open(sample['filepath']) as original:
            image = original.convert('RGB')
        scale = min(720 / image.width, 440 / image.height)
        size = (round(image.width * scale), round(image.height * scale))
        thumbnail = image.resize(size)
        canvas = Image.new('RGB', (size[0] * 2, size[1] + 34), '#101820')
        draw = ImageDraw.Draw(canvas)
        for panel, (name, payload, color) in enumerate([
            ('Ground truth', sample['target'], '#45df91'),
            (f'Predictions (score ≥ {confidence:g})', sample['prediction'], '#ffbb54')]):
            offset = panel * size[0]
            canvas.paste(thumbnail, (offset, 34))
            draw.text((offset + 8, 7), name, fill=color, font=font)
            count = 0
            for i, (box, label) in enumerate(zip(payload['boxes'], payload['labels'])):
                score = payload.get('scores', [1.] * len(payload['boxes']))[i]
                if panel and score < confidence:
                    continue
                if panel and count >= 30:
                    break
                count += 1
                x1, y1, x2, y2 = [v * scale for v in box]
                x1, x2, y1, y2 = x1 + offset, x2 + offset, y1 + 34, y2 + 34
                draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
                label_text = classes[int(label)] if 0 <= int(label) < len(classes) else str(label)
                text = label_text + (f' {score:.2f}' if panel else '')
                draw.text((x1, max(34, y1 - 15)), text, fill=color, font=font, stroke_width=1, stroke_fill='#000000')
        path = output / f'sample-{index}.jpg'
        temporary = path.with_suffix('.tmp')
        canvas.save(temporary, format='JPEG', quality=85)
        temporary.replace(path)
        metadata.append({'image_id': sample['image_id'], 'filepath': sample['filepath'],
                         'image': path.name, 'progress': progress, 'confidence_threshold': confidence,
                         'predictions': sample['prediction'], 'ground_truth': sample['target']})
    temporary = output / 'metadata.json.tmp'
    temporary.write_text(json.dumps(metadata, indent=2))
    temporary.replace(output / 'metadata.json')


def rfdetr_previews(model, coco_dir, image_root, output, classes, epoch):
    """RF-DETR's epoch callback runs after validation, using four fixed val frames."""
    data = load_coco(coco_dir, 'val', image_root, max_samples=4)
    samples = []
    for im in data['images']:
        result = model.predict(im['filepath'], threshold=0.3)
        annotations = clipped_annotations(im, data['annotations'][im['id']])
        targets = {'boxes': [[x, y, x+w, y+h] for x, y, w, h in [a['bbox'] for a in annotations]],
                   'labels': [data['category_to_index'][a['category_id']] for a in annotations]}
        samples.append({'image_id': im['id'], 'filepath': im['filepath'], 'target': targets,
            'prediction': {'boxes': result.xyxy.tolist(), 'scores': result.confidence.tolist(),
                           'labels': result.class_id.tolist()}})
    save_previews(output, samples, classes, {'epoch': epoch})


def write_model_info(output, model, classes, recipe):
    info = {'parameters': sum(parameter.numel() for parameter in model.parameters()),
            'trainable_parameters': sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            'classes': classes, 'recipe': recipe}
    path = Path(output) / 'model_info.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(info, indent=2))
    temporary.replace(path)




def detection_metric():
    from torchmetrics.detection import MeanAveragePrecision
    metric = MeanAveragePrecision(box_format='xyxy', class_metrics=True, backend='faster_coco_eval')
    metric.warn_on_many_detections = False
    return metric


class RFCocoEvaluator:
    def __init__(self, coco_gt, iou_types):
        from faster_coco_eval import COCO
        assert tuple(iou_types) == ('bbox',)
        self.coco_gt = COCO(coco_gt.dataset)
        self.predictions = {}
        self.coco_eval = {}

    def update(self, predictions):
        self.predictions.update({i: {k:v.detach().cpu() for k,v in p.items()} for i,p in predictions.items()})

    def synchronize_between_processes(self):
        pass

    def accumulate(self):
        from faster_coco_eval import COCOeval_faster
        from torchvision.ops import box_convert
        results = []
        for image_id, pred in self.predictions.items():
            boxes = box_convert(pred['boxes'].float(), 'xyxy', 'xywh').tolist()
            results.extend({'image_id':image_id,'category_id':int(label),'bbox':box,'score':float(score)}
                           for box,label,score in zip(boxes,pred['labels'],pred['scores']))
        evaluation = COCOeval_faster(self.coco_gt,self.coco_gt.loadRes(results),'bbox')
        evaluation.params.imgIds = list(self.predictions)
        self.predictions.clear()
        evaluation.evaluate()
        evaluation.accumulate()
        self.coco_eval['bbox'] = evaluation

    def summarize(self):
        self.coco_eval['bbox'].summarize()


class CocoDataset:
    def __init__(self, coco_dir, split, image_root=None, augment=False, processor=None, uint8=False):
        self.data = load_coco(coco_dir, split, image_root)
        self.augment, self.processor = augment, processor
        self.uint8 = uint8

    def __len__(self):
        return len(self.data['images'])

    def __getitem__(self, index):
        import torch
        from PIL import Image
        from torchvision.transforms import ColorJitter, functional as F
        im = self.data['images'][index]
        image = Image.open(im['filepath']).convert('RGB')
        anns = clipped_annotations(im, self.data['annotations'][im['id']])
        if self.processor is not None:
            anns = [{**a, 'category_id': self.data['category_to_index'][a['category_id']]} for a in anns]
            encoded = self.processor(images=image, annotations={'image_id': im['id'], 'annotations': anns}, return_tensors='pt')
            return {k: v[0] for k, v in encoded.items()}
        boxes = torch.tensor([a['bbox'] for a in anns], dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        labels = torch.tensor([self.data['category_to_index'][a['category_id']] + 1 for a in anns], dtype=torch.int64)
        if self.augment:
            if torch.rand(1).item() < 0.5:
                image = F.hflip(image)
                boxes[:, [0, 2]] = im['width'] - boxes[:, [2, 0]]
            image = ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2))(image)
        return (F.pil_to_tensor(image) if self.uint8 else F.to_tensor(image)), {'boxes': boxes, 'labels': labels,
            'image_id': torch.tensor(im['id']), 'area': torch.tensor([a['area'] for a in anns]),
            'iscrowd': torch.tensor([a.get('iscrowd', 0) for a in anns], dtype=torch.int64)}


def collate_images(batch):
    return tuple(zip(*batch))


def train_torchvision(family, args, output, classes):
    import torch
    from torch.utils.data import DataLoader
    wrapper = load_pretrained(family, args.device)
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(output / 'tensorboard'))
    model = replace_torchvision_head(wrapper._model, family, len(classes)).to(args.device)
    if args.channels_last:
        model.to(memory_format=torch.channels_last)
    steps = args.max_steps or 50000
    recipe = {'batch_size': args.batch_size or 4, 'lr': args.lr or .005, 'max_steps': steps,
              'milestones': [int(.6*steps), int(.85*steps)], 'warmup_steps': max(1,int(args.warmup_ratio*steps)), 'eval_period': args.eval_every,
              'momentum': 0.9, 'weight_decay': 0.0001,
              'grad_accum_steps': args.grad_accum_steps or 1, 'amp': args.amp, 'precision': args.precision}
    recipe['eval_score_threshold'] = model.score_thresh if family == 'retinanet' else model.roi_heads.score_thresh
    dtype = torch.bfloat16 if args.precision == 'bf16' else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp and args.precision == 'fp16')
    write_model_info(output, model, classes, recipe)
    print(f'Training recipe: {recipe}', flush=True)
    # Torchvision's default short-edge resize (800, max 1333) matches Detectron2's default.
    train_data = CocoDataset(args.coco_dir, 'train', args.image_root, augment=True, uint8=True)
    val_data = CocoDataset(args.coco_dir, 'val', args.image_root, uint8=True)
    train_loader = DataLoader(train_data, batch_size=recipe['batch_size'], shuffle=True,
        collate_fn=collate_images, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0,
        prefetch_factor=1 if args.workers else None)
    val_loader = DataLoader(val_data, batch_size=recipe['batch_size'], collate_fn=collate_images,
        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0,
        prefetch_factor=1 if args.workers else None)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=recipe['lr'],
        momentum=recipe['momentum'], weight_decay=recipe['weight_decay'])
    def lr_factor(step):
        warmup = 0.001 + (1 - 0.001) * min(step / recipe['warmup_steps'], 1)
        return warmup * 0.1 ** sum(step >= m for m in recipe['milestones'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    start, best, stale = 0, -1.0, 0
    if args.resume:
        state = torch.load(args.resume, map_location=args.device, weights_only=True)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        if 'scaler' in state:
            scaler.load_state_dict(state['scaler'])
        start, best = state['step'], state['best_map']
    iterator = iter(train_loader)
    for step in range(start, recipe['max_steps']):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.
        for micro_step in range(recipe['grad_accum_steps']):
            try:
                images, targets = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                images, targets = next(iterator)
            images = [im.to(args.device, non_blocking=True).float().div_(255) for im in images]
            targets = [{k: v.to(args.device, non_blocking=True) for k, v in t.items()} for t in targets]
            with torch.autocast('cuda', dtype=dtype, enabled=args.amp and args.device == 'cuda'):
                loss = sum(model(images, targets).values()) / recipe['grad_accum_steps']
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite training loss at step {step+1}: {loss.item()}')
            scaler.scale(loss).backward()
            step_loss += loss.item()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if (step+1) % 20 == 0:
            print(f'Step {step+1}: loss={step_loss:.5f}', flush=True)
            epoch = (step+1) * recipe['batch_size'] * recipe['grad_accum_steps'] / len(train_data)
            append_history(output, {'step': step+1, 'epoch': epoch, 'train_loss': step_loss,
                'learning_rate': optimizer.param_groups[0]['lr']})
            writer.add_scalar('train/loss', step_loss, step+1)
            writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], step+1)
        if (step+1) % recipe['eval_period'] == 0 or step+1 == recipe['max_steps']:
            model.eval()
            metric = detection_metric()
            previews = []
            seen = 0
            with torch.no_grad():
                for images, targets in val_loader:
                    with torch.autocast('cuda', dtype=dtype, enabled=args.amp and args.device == 'cuda'):
                        predictions = model([im.to(args.device, non_blocking=True).float().div_(255) for im in images])
                    cpu_predictions = [{k: v.float().cpu() if v.is_floating_point() else v.cpu()
                                        for k, v in p.items()} for p in predictions]
                    metric.update(cpu_predictions, list(targets))
                    for i, prediction in enumerate(cpu_predictions):
                        if len(previews) < 4:
                            im = val_data.data['images'][seen + i]
                            previews.append({'image_id': im['id'], 'filepath': im['filepath'],
                                'target': {'boxes': targets[i]['boxes'].tolist(), 'labels': (targets[i]['labels'] - 1).tolist()},
                                'prediction': {'boxes': prediction['boxes'].tolist(), 'scores': prediction['scores'].tolist(),
                                               'labels': (prediction['labels'] - 1).tolist()}})
                    seen += len(targets)
            validation = map_metrics(metric.compute(), classes, label_offset=1)
            score = validation['map']
            epoch = (step+1) * recipe['batch_size'] * recipe['grad_accum_steps'] / len(train_data)
            append_history(output, {'step': step+1, 'epoch': epoch,
                **{f'val_{key}': value for key, value in validation.items()}})
            for key, value in validation.items():
                writer.add_scalar(f'val/{key}', value, step+1)
            writer.flush()
            save_previews(output, previews, classes, {'step': step+1, 'epoch': epoch})
            improved = score > best
            stale = 0 if improved else stale+1
            best = max(best, score)
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(), 'step': step+1, 'best_map': best}
            torch.save(state, output / 'last.pth')
            if improved:
                torch.save(state, output / 'best.pth')
            print(f'Step {step+1}: validation mAP={score:.5f}', flush=True)
            if stale >= args.patience:
                break
    if not (output / 'best.pth').exists():
        raise FileNotFoundError('No best.pth; resume into the original output directory or increase --max-steps')
    writer.close()
    return output / 'best.pth', recipe


class TransformerCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        # Some processors (e.g. RT-DETR) return fixed-size images without a mask.
        import torch
        values = [item['pixel_values'] for item in batch]
        if all(v.shape == values[0].shape for v in values):
            result = {'pixel_values': torch.stack(values)}
            if 'pixel_mask' in batch[0]:
                result['pixel_mask'] = torch.stack([item['pixel_mask'] for item in batch])
        else:
            result = dict(self.processor.pad(values, return_tensors='pt'))
        result['labels'] = [item['labels'] for item in batch]
        return result


def train_transformer(family, args, output, classes):
    from transformers import (AutoImageProcessor, AutoModelForObjectDetection, Trainer,
                              TrainingArguments, EarlyStoppingCallback)
    from transformers import TrainerCallback
    import torch
    from torchvision.ops import box_convert
    source = MODELS[family].source
    # Use the Zoo's matching HF configuration; changing class count replaces the head.
    wrapper = load_pretrained(family, args.device)
    labels = dict(wrapper._model.config.id2label)
    old_heads = {k:v.detach().cpu() for k,v in wrapper._model.state_dict().items()
                 if any(s in k for s in ['class_embed','enc_score_head','class_labels_classifier'])} if args.semantic_init else {}
    config = wrapper._model.config
    config.id2label = dict(enumerate(classes))
    config.label2id = {name: i for i, name in enumerate(classes)}
    del wrapper
    model = AutoModelForObjectDetection.from_pretrained(source, config=config, ignore_mismatched_sizes=True)
    if args.semantic_init:
        groups = {'Vehicle':{'car','bus','truck'}, 'VulnerableVehicle':{'bicycle','motorcycle','motorbike'}, 'Pedestrian':{'person'}}
        indices = [[i for i,label in labels.items() if label.lower() in groups[name]] for name in classes]
        assert all(indices)
        state = model.state_dict()
        for name,old in old_heads.items():
            new = state[name]
            if old.shape != new.shape:
                rows = torch.stack([old[ids].mean(0) for ids in indices])
                if new.shape[0] == len(classes)+1:
                    rows = torch.cat([rows,old[-1:]])
                state[name] = rows
        model.load_state_dict(state)
        print('COCO class initialization:',dict(zip(classes,indices)),flush=True)
    if args.channels_last:
        model.to(memory_format=torch.channels_last)
    processor = AutoImageProcessor.from_pretrained(source)
    # Keep each processor's expected sizing schema (DETR vs real-time DETRs).
    processor.size = ({'shortest_edge': 512, 'longest_edge': 864}
                      if 'shortest_edge' in processor.size else {'height': 512, 'width': 512})
    recipe = {'epochs': args.epochs or 30, 'batch_size': args.batch_size or 4,
              'grad_accum_steps': args.grad_accum_steps or 4, 'lr': args.lr or 1e-4, 'backbone_lr': args.backbone_lr,
              'patience': args.patience, 'resolution': 512, 'amp': args.amp, 'precision': args.precision,
              'semantic_init': args.semantic_init, 'channels_last': args.channels_last,
              'selection_metric': 'validation_map'}
    write_model_info(output, model, classes, recipe)
    training_args = TrainingArguments(output_dir=str(output), num_train_epochs=recipe['epochs'],
        per_device_train_batch_size=recipe['batch_size'], per_device_eval_batch_size=recipe['batch_size'],
        gradient_accumulation_steps=recipe['grad_accum_steps'], learning_rate=recipe['lr'], dataloader_num_workers=args.workers,
        dataloader_pin_memory=True, dataloader_persistent_workers=args.workers > 0, warmup_ratio=args.warmup_ratio,
        logging_steps=10, optim='adamw_torch_fused' if args.device == 'cuda' else 'adamw_torch',
        max_steps=args.max_steps or -1, seed=args.seed, data_seed=args.seed,
        fp16=args.amp and args.precision == 'fp16' and args.device == 'cuda',
        bf16=args.amp and args.precision == 'bf16' and args.device == 'cuda', tf32=args.device == 'cuda', use_cpu=args.device == 'cpu',
        eval_strategy='epoch', save_strategy='epoch', save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model='eval_map', greater_is_better=True,
        remove_unused_columns=False, eval_do_concat_batches=False, prediction_loss_only=True,
        report_to=['tensorboard'])
    if args.device.startswith('cuda:'):
        import torch
        torch.cuda.set_device(int(args.device.split(':')[1]))
    val_dataset = CocoDataset(args.coco_dir, 'val', args.image_root, processor=processor)

    class HistoryCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                append_history(output, {'step': state.global_step, 'epoch': state.epoch, **logs})

    class DetectionTrainer(Trainer):
        def create_optimizer(self):
            if self.optimizer is None:
                groups = [{'params': [p for n, p in self.model.named_parameters() if p.requires_grad and ('backbone' in n) == backbone],
                           'lr': recipe['backbone_lr'] if backbone else recipe['lr']} for backbone in [True, False]]
                self.optimizer = torch.optim.AdamW(groups, weight_decay=1e-4, fused=args.device == 'cuda')
            return self.optimizer

        # Stream detections into COCO evaluation rather than retaining decoder
        # tensors for the entire validation split in Trainer's prediction buffer.
        def evaluation_loop(self, *loop_args, **loop_kwargs):
            self.detection_metric = detection_metric()
            self.preview_samples = []
            self.validation_seen = 0
            result = super().evaluation_loop(*loop_args, **loop_kwargs)
            scores = map_metrics(self.detection_metric.compute(), classes)
            metrics = {**result.metrics, **{f'eval_{key}': value for key, value in scores.items()}}
            save_previews(output, self.preview_samples, classes,
                          {'step': self.state.global_step, 'epoch': self.state.epoch})
            del self.detection_metric
            return result._replace(metrics=metrics)

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            inputs = self._prepare_inputs(inputs)
            with torch.no_grad(), self.compute_loss_context_manager():
                outputs = model(**inputs)
            labels = inputs['labels']
            sizes = torch.stack([target['orig_size'] for target in labels])
            predictions = processor.post_process_object_detection(outputs, target_sizes=sizes, threshold=0.)
            targets = []
            for target, size in zip(labels, sizes):
                h, w = size.unbind()
                scale = torch.stack([w, h, w, h])
                boxes = box_convert(target['boxes'], 'cxcywh', 'xyxy') * scale
                targets.append({'boxes': boxes.cpu(), 'labels': target['class_labels'].cpu()})
            predictions = [{k: v.detach().float().cpu() if k != 'labels' else v.detach().cpu()
                            for k, v in prediction.items()} for prediction in predictions]
            self.detection_metric.update(predictions, targets)
            for i, prediction in enumerate(predictions):
                if len(self.preview_samples) < 4:
                    im = val_dataset.data['images'][self.validation_seen + i]
                    self.preview_samples.append({'image_id': im['id'], 'filepath': im['filepath'],
                        'target': {k: v.tolist() for k, v in targets[i].items()},
                        'prediction': {k: v.tolist() for k, v in prediction.items()}})
            self.validation_seen += len(targets)
            return outputs.loss.detach().mean(), None, None

    trainer = DetectionTrainer(model=model, args=training_args,
        train_dataset=CocoDataset(args.coco_dir, 'train', args.image_root, processor=processor),
        eval_dataset=val_dataset,
        data_collator=TransformerCollator(processor), callbacks=[HistoryCallback(), EarlyStoppingCallback(early_stopping_patience=args.patience)])
    trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
    checkpoint = output / 'best'
    trainer.save_model(str(checkpoint))
    processor.save_pretrained(str(checkpoint))
    return checkpoint, recipe


if __name__ == '__main__':
    main()
