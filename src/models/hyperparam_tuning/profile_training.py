"""Measure training throughput on real ZOD tensors, excluding image-loading time."""
import argparse
import sys
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torchvision.ops import box_convert

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.models.adapters import load_pretrained, replace_torchvision_head
from src.models.catalog import MODELS
from src.models.train import CocoDataset, TransformerCollator


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('family', choices=MODELS)
    p.add_argument('--coco-dir', type=Path, required=True)
    p.add_argument('--batch-size', type=int, required=True)
    p.add_argument('--precision', choices=['fp32', 'fp16', 'bf16'], required=True)
    p.add_argument('--steps', type=int, default=10)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--channels-last', action='store_true')
    p.add_argument('--rf-groups', type=int, default=13)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(43)
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    family, batch = args.family, args.batch_size
    backend = MODELS[family].backend
    dtype = torch.bfloat16 if args.precision == 'bf16' else torch.float16
    wrapper = load_pretrained(family, 'cuda')
    if backend == 'transformers':
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
        config = wrapper._model.config
        config.id2label = dict(enumerate(['Vehicle', 'VulnerableVehicle', 'Pedestrian']))
        config.label2id = {v: k for k, v in config.id2label.items()}
        del wrapper
        model = AutoModelForObjectDetection.from_pretrained(MODELS[family].source, config=config, ignore_mismatched_sizes=True).cuda()
        processor = AutoImageProcessor.from_pretrained(MODELS[family].source)
        processor.size = {'shortest_edge': 512, 'longest_edge': 864} if 'shortest_edge' in processor.size else {'height': 512, 'width': 512}
        dataset = CocoDataset(args.coco_dir, 'train', processor=processor)
        inputs = TransformerCollator(processor)([dataset[i % 4] for i in range(batch)])
        inputs = {k: [{a: b.cuda() for a, b in t.items()} for t in v] if k == 'labels' else v.cuda() for k, v in inputs.items()}
        def loss_fn():
            return model(**inputs).loss
    else:
        dataset = CocoDataset(args.coco_dir, 'train', uint8=backend == 'torchvision')
        samples = [dataset[i % 4] for i in range(batch)]
        if backend == 'torchvision':
            model = replace_torchvision_head(wrapper._model, family, 3).cuda()
            images = [im.cuda().float().div_(255) for im, _ in samples]
            targets = [{k: v.cuda() for k, v in t.items()} for _, t in samples]
            def loss_fn():
                return sum(model(images, targets).values())
        else:
            size = 560 if backend == 'rfdetr' else 512
            images = torch.stack([F.interpolate(im[None], (size, size), mode='bilinear', align_corners=False)[0] for im, _ in samples]).cuda()
            targets = []
            for im, target in samples:
                h, w = im.shape[-2:]
                targets.append({'boxes': (box_convert(target['boxes'], 'xyxy', 'cxcywh') / torch.tensor([w,h,w,h])).cuda(),
                                'labels': (target['labels']-1).cuda()})
            if backend == 'rfdetr':
                from rfdetr.models import build_criterion_and_postprocessors
                from rfdetr.util.misc import NestedTensor
                native = wrapper._model
                if args.rf_groups != 13:
                    config = native.model_config.model_dump()
                    config['group_detr'] = args.rf_groups
                    native = type(native)(**config)
                    del wrapper
                native.model.reinitialize_detection_head(3)
                model = native.model.model.cuda()
                native.model.args.num_classes = 3
                criterion, _ = build_criterion_and_postprocessors(native.model.args)
                images = (images-torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None])/torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
                nested = NestedTensor(images, torch.zeros((batch,size,size),device='cuda',dtype=torch.bool))
                def loss_fn():
                    losses = criterion(model(nested, targets), targets)
                    return sum(v*criterion.weight_dict[k] for k,v in losses.items() if k in criterion.weight_dict)
            else:
                from ultralytics import YOLO
                from ultralytics.cfg import get_cfg
                model = YOLO('models/yolo/before_tuning/last.pt').model.float().cuda()
                model.args = get_cfg()
                inputs = {'img': images, 'cls': torch.cat([t['labels'] for t in targets]).float()[:,None],
                          'bboxes': torch.cat([t['boxes'] for t in targets]),
                          'batch_idx': torch.cat([torch.full((len(t['labels']),),i,device='cuda') for i,t in enumerate(targets)])}
                def loss_fn():
                    return model(inputs)[0].sum()
    model.train()
    if backend == 'ultralytics':
        model.requires_grad_(True)
        for name, parameter in model.named_parameters():
            if '.dfl.' in name:
                parameter.requires_grad_(False)
    if args.channels_last:
        if backend == 'ultralytics':
            inputs['img'] = inputs['img'].contiguous(memory_format=torch.channels_last)
        else:
            model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=args.precision == 'fp16')
    torch.cuda.reset_peak_memory_stats()
    losses, durations = [], []
    for step in range(args.steps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=dtype, enabled=args.precision != 'fp32'):
            loss = loss_fn()
        assert torch.isfinite(loss), loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        durations.append(time.perf_counter()-start)
        losses.append(float(loss.detach()))
    result = {'family': family, 'batch_size': batch, 'precision': args.precision,
              'channels_last': args.channels_last, 'images_per_second': batch/(sum(durations[3:])/len(durations[3:])),
              'peak_gib': torch.cuda.max_memory_allocated()/1024**3, 'losses': losses,
              'seconds_per_batch': durations, 'note': 'Real-image GPU compute probe; excludes image loading and validation'}
    result['rf_groups'] = args.rf_groups if backend == 'rfdetr' else None
    result['trainable_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
