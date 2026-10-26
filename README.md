# ZOD detector pipeline

Load detectors with the FiftyOne Model Zoo, fine-tune on the prepared Zenseact
COCO splits, and apply trained checkpoints through `Dataset.apply_model()`.
The catalog follows the closed-set families in
[OD_demand_scales](https://github.com/femartip/OD_demand_scales/blob/main/scripts/inference/get_predictions.py),
selecting one representative per family rather than every size/generation.

| Family alias | FiftyOne Zoo model | Training backend |
| --- | --- | --- |
| `yolo` | `yolo26l-coco-torch` | Ultralytics |
| `rf-detr` | `rfdetr-large-coco-torch` | Existing RF-DETR fork |
| `rt-detr` | `rtdetr-v2-l-coco-torch` | Transformers |
| `d-fine` | `dfine-xlarge-coco-torch` | Transformers |
| `detr` | `detection-transformer-torch` | Transformers |
| `faster-rcnn` | `faster-rcnn-resnet50-fpn-coco-torch` | Torchvision |
| `retinanet` | `retinanet-resnet50-fpn-coco-torch` | Torchvision |

These are explicit, reproducible detector selections, checked on October 23, 2026. YOLO26 replaces YOLO11, and RT-DETR-v2 replaces RT-DETR-v1.
Faster R-CNN and RetinaNet use the Torchvision variants exposed by the Zoo;
these entries are not Torchvision's separate `_v2` architectures. InternImage
remains historical source under `src/models/internimage/`; it is outside this
closed-set Zoo panel.

## Environment

Use the root Poetry environment with Python 3.11 or 3.12:

```bash
poetry env use python3.11
poetry install
poetry run python src/models/run_inference.py --list-models
```

All dependencies are declared in the root `pyproject.toml` and locked in
`poetry.lock`. FiftyOne is pinned to 1.16 so it can share Pillow 11 with the
existing AutoGluon and Trackio versions. RT-DETR-v2 loads its configured
`PekingU/rtdetr_v2_r101vd` checkpoint through the generic Transformers Zoo entry
in this version; the detector architecture is unchanged. The GraphQL dependency
is constrained to the compatible 3.2 series.

PyTorch and Torchvision use CUDA 12.8 builds. The RTX A5000 has a working
595.99.02 NVIDIA driver. Workspace sandbox processes do not expose GPU device
nodes, so training must run in a host terminal or with GPU access enabled.
Use `poetry run python -c "import torch; print(torch.cuda.is_available())"` in
that execution context to verify access. Select a GPU with
`CUDA_VISIBLE_DEVICES` when needed.

FiftyOne requires MongoDB for dataset inference. On systems unsupported by its
bundled installer, install MongoDB separately and set `FIFTYONE_DATABASE_URI`,
for example `mongodb://localhost:27017`.

## Existing data

Default annotations: `data/zod_coco/*_{train,val,test}.json`. The current files
contain three classes in this order: Vehicle, VulnerableVehicle, Pedestrian.
The dataset resolver reuses image paths under this project or the sibling
`../Validators-OD/data/zod/`. Relocate images explicitly with
`--image-root /path/to/zod` (the directory containing `single_frames`).

The prepared geographic splits and annotation filtering are retained. Training
validates that split IDs do not overlap and class definitions agree. Validation
selects checkpoints; test data is reserved for inference. Sparse COCO category
IDs are mapped explicitly to each backend's indexing convention. Native YOLO
and RF-DETR label files are generated under the model's output directory and
images are symlinked, without re-downloading or copying the image dataset.

## Fine-tuning

`src/models/train.py` contains the detector training backends, validation AP,
and saved training diagnostics. Shared COCO loading and dataset preparation live
in `src/data/dataset.py`, used by detectors and assessors. `finetune_all.py`
queues detectors; `adapters.py` loads models and adjusts their classification
heads to the ZOD classes.

```bash
poetry run python src/models/train.py yolo
poetry run python src/models/train.py rf-detr
poetry run python src/models/train.py rt-detr
poetry run python src/models/train.py d-fine
poetry run python src/models/train.py detr
poetry run python src/models/train.py faster-rcnn
poetry run python src/models/train.py retinanet
```

Each command loads its Zoo model and trains with the corresponding backend;
FiftyOne's Zoo wrappers provide inference rather than a common training API.
Artifacts go to `models/<family>/` and include a `manifest.json`
recording class order, model identity, checkpoint, library versions, and settings.
New runs refuse to replace an artifact manifest; use a new `--output-dir` or
`--resume /path/to/backend/checkpoint` in the original output directory.

Default recipes (measured tuning overrides are described below):

- **YOLO:** up to 50 epochs, batch 32, AMP, 512 input, seed 43, patience 5,
  rectangular batches and multiscale 0.25. Set `--optimizer` explicitly when
  tuning `--lr`; the native automatic optimizer can replace the requested LR.
  Training reloads an unfused checkpoint from the Zoo wrapper.
- **RF-DETR:** up to 30 epochs, batch 4, accumulation 4, head LR 1e-4,
  encoder LR 1e-5, native BF16 AMP, patience 5, no multiscale/expanded scales/EMA,
  and no automatic test run. `--rf-groups` controls training-only query groups.
  Resolution is corrected from 512 to **560**: Large's 14-pixel patches and four
  windows require a multiple of 56. Labels use the standard RF-DETR directory
  layout rather than the fork's special ZOD paths.
- **Faster R-CNN/RetinaNet:** SGD with momentum 0.9 and weight decay 1e-4,
  AMP, configurable microbatch/accumulation, up to 50,000 steps, 5% warmup,
  LR drops at 60%/85% of the budget and validation early stopping. Images remain
  uint8 in pinned CPU queues and convert to float on the GPU. Horizontal flips
  and brightness/contrast/saturation jitter are retained.
  Validation COCO mAP selects `best.pth`. Faster R-CNN now uses ResNet-50 FPN,
  replacing the old Detectron2 ResNet-101 FPN; old `.pth` files are not compatible.
- **Transformers:** up to 30 epochs, 512 input, seed 43 and patience 5.
  Fused AdamW uses separate backbone/head learning rates, AMP, 5% warmup and
  persistent data-loader workers. Validation detection AP selects checkpoints and
  drives early stopping; validation loss is also recorded. `--semantic-init` can
  initialize replacement class heads from the corresponding COCO class weights.

`--epochs`, `--batch-size`, `--grad-accum-steps`, `--workers`, `--seed`, and
`--device` are configurable. Torchvision uses `--max-steps` instead of `--epochs`.
YOLO manages gradient accumulation internally.

The original full-data YOLO run is stopped, with its last/best checkpoints saved
under `models/yolo/before_tuning/`. Short tuning pilots never launch final training:

```bash
poetry run python src/models/hyperparam_tuning/tune.py
poetry run python src/models/hyperparam_tuning/tune.py --models d-fine detr rt-detr --phase pilots \
  --semantic-init --compare-existing --epochs 6 --rates 0.0001 --backbone-lr 0.000001
poetry run python src/models/hyperparam_tuning/tune_rf_groups.py
poetry run python src/models/hyperparam_tuning/tune_channels.py
poetry run python src/models/hyperparam_tuning/tuning_report.py
```

Tuning uses fixed random subsets of 768 train/192 validation images, seed 43,
and no test images. GPU probes compare precision, batch sizes and channels-last
layout, followed by short learning-rate/initialization pilots. RF-DETR query-group
pilots retain settings within 0.02 AP of the best short pilot and prefer throughput.
Probes omit data loading and use SGD, so their
throughput is not an end-to-end training ETA. Curves and measurements are in
`reports/tuning/index.html`; selected settings are in `data/tuning/recipes.json`.
YOLO pilots warm-start from the saved ZOD checkpoint; others start from COCO.
Eight workers are also compared with two/four using a separate image-loader probe.

When ready, launch the final panel yourself:

```bash
poetry run python src/models/finetune_all.py \
  --recipes data/tuning/recipes.json --artifacts-dir models/tuned --log-dir logs/finetune-tuned
```

The selected YOLO recipe starts a new optimizer schedule from its preserved
checkpoint. `--init-checkpoint` loads weights; `--resume` restores optimizer state.
Final outputs are separate from pilots and the original run. The queue skips
completed manifests and stops on failure. Logs/status go to the selected log
directory; tracking runs offline. Model inference accepts `--artifacts-dir models/tuned`.

## Inference and metrics

```bash
# Evaluate one fine-tuned model on test (default)
poetry run python src/models/run_inference.py yolo

# Use an artifact from a custom training output directory
poetry run python src/models/run_inference.py yolo /path/to/artifact-directory

# Evaluate several fine-tuned models on validation
poetry run python src/models/run_inference.py --models yolo rf-detr d-fine --split val

# Evaluate all seven fine-tuned artifacts
poetry run python src/models/run_inference.py all

# Explicit coarse-mapped COCO baseline, without fine-tuning
poetry run python src/models/run_inference.py yolo --pretrained --max-samples 10 --device cpu
```

Family aliases and full Zoo names are accepted. `--train`, `--validate`, and
`--test` also select splits, and may be combined. Every model/split gets a fresh
temporary FiftyOne dataset. Failed or missing predictions abort that result;
temporary datasets are deleted afterward.

Fine-tuned test outputs retain the existing feature-merger paths:

- `results/<family>/detections.json`: per-image `iou`, `lrp` (the existing
  **1 − LRP error** score, so higher is better), and confidence values.
- `results/<family>/predictions.json`: boxes, scores, and zero-based ZOD labels.
- `results/<family>/predictions.meta.json`: model, artifact, split, class order,
  coordinate space, sample count and confidence threshold.

Boxes retain the legacy 1344×768 `xyxy` output space; models perform their own
native preprocessing on source images. The existing metric implementation is
reused, with corrected matching of alternative unmatched ground-truth boxes and
zero IoU contributions for missed objects in empty-prediction images. Both-empty images
score 1 by convention. Prediction confidence defaults to 0.5 across models; use
`--confidence-threshold` to change it. `--discretize-threshold` additionally writes
`detections_disc.json`. Train/validation outputs have their own split subdirectory.
Partial runs have an additional `partial-N` subdirectory, so they cannot replace
complete outputs.

Pretrained baselines are saved separately under
`results/pretrained/<family>/<split>/`. Their coarse mapping is car/bus/truck →
Vehicle, bicycle/motorcycle → VulnerableVehicle, person → Pedestrian; other COCO
classes are omitted. This is an approximate baseline, not a ZOD-fine-tuned model.

The old model-specific pre-NMS `--save-raw` and zero-IoU image-export options are
not provided by this common FiftyOne path. Final detection containers do not expose
full class-probability vectors; existing raw prediction artifacts remain historical.

## Learning curves and validation examples

The complete panel runs sequentially on the single RTX A5000. A pending model
has no training results yet; COCO-pretrained preflight checks are separate from
ZOD fine-tuning measurements.

YOLO's `results.csv` and RF-DETR's native `log.txt` supply epoch metrics.
Torchvision writes `learning_history.jsonl` and TensorBoard scalars for training
loss, validation mAP/AP50/AP75/recall, and per-class AP. Transformers additionally
streams detections into full-split COCO evaluation while computing validation
loss, without retaining full decoder tensors. Validation AP selects checkpoints
for the current adapters. Loss magnitudes should be compared within a model, not across
families. Per-class AP is saved where the backend exports it.

Check for improving validation AP together with decreasing training loss. One
validation point does not establish a trend. The test split is excluded from
training diagnostics and checkpoint selection.

## Image-assessor study

Predict a frozen detector's per-image performance from the image alone. Detector
outputs and ground truth construct training targets; neither is an assessor input.

- **Targets:** retain the existing mean matched IoU with zeros for unmatched
  predictions/objects, and compare it with `1 - LRP error` (both higher is better).
  LRP combines normalized localization error, false positives and false negatives
  at an explicit IoU matching threshold; it is not universally preferable to the
  existing score. Audit matching, empty-image cases and ignored annotations first.
  Select confidence thresholds using detector validation data and freeze them.
- **Splits:** freeze detector checkpoints, then audit prior use of the 9,667
  detector-test images before repurposing them into approximately 60/20/20
  assessor train/validation/test splits, grouped by geography or drive. Save one
  split for all methods; fit preprocessing only on assessor training data. This
  repurposes the original test pool rather than creating a new independent set.
- **Encoders:** ImageNet-pretrained ResNet-18, ResNet-50, MobileNetV3-Small and
  ConvNeXt-Tiny; DINOv2 ViT-B/14, DINOv3 ViT-B/16, SigLIP 2 Base NaFlex and MAE
  ViT-Base. Pin weight versions; DINOv3 requires model access acceptance.
- **Comparison:** screen frozen global features with ridge regression and a
  small MLP, then compare spatial features using grid pooling. Use full-frame,
  aspect-preserving preprocessing and test resolution separately. Partially and
  fully fine-tune shortlisted encoders on performance scores, keeping their heads
  fixed in design. Start with one completed detector, then expand across families
  and compare separate assessors with a shared encoder and detector-specific heads.
- **Evaluation:** report score MAE, R², Spearman correlation, risk–coverage curves,
  training/validation curves, predicted-versus-measured plots and label-budget
  learning curves. Include constant and image-quality baselines, repeat shortlisted
  runs with multiple seeds, and estimate uncertainty by geographic/drive groups.

The current three-class annotation filters define the study's scope. Difficulty-
changing augmentations require recomputed detector targets. New JEPA objectives
are deferred until these existing representations have been evaluated.

Metric references: [LRP, ECCV 2018](https://arxiv.org/abs/1807.01696) and
[extended TPAMI study](https://arxiv.org/abs/2011.10772).

The implementation is under `src/assessors/`. Run from the repository root:

```bash
# Generate predictions from completed detector artifacts, retaining low scores for calibration
poetry run python src/models/run_inference.py yolo --validate --test --confidence-threshold 0.01

# Build one reusable geographic split and targets; supply all detector files when rebuilding targets
poetry run python -m src.assessors.prepare \
  --groups ../Validators-OD/data/metafeatures.csv \
  --predictions yolo=results/yolo/predictions.json \
  --calibration yolo=results/yolo/val/predictions.json

# Cache features; add the remaining encoder aliases when ready
poetry run python -m src.assessors.features --encoders resnet18 resnet50 convnext_tiny

# Screen both target scores, constant/image-quality baselines, ridge and MLP assessors
poetry run python -m src.assessors.run --detectors yolo \
  --features data/assessors/features/224x448-grid2x4 \
  --encoders resnet18 resnet50 convnext_tiny

# Fine-tune a shortlisted encoder, keeping the MLP architecture unchanged
poetry run python -m src.assessors.train --encoder resnet18 --mode partial \
  --pool spatial --detectors yolo --target iou --seeds 43 44 45 --fractions 0.1 0.25 0.5 1
poetry run python -m src.assessors.run --report-only
```

Encoder aliases: `resnet18`, `resnet50`, `mobilenet_v3_small`, `convnext_tiny`,
`dinov2`, `dinov3`, `siglip2`, `mae`. Feature extraction stores global, mean-patch
and spatial features plus image-quality descriptors. `--size 448 896` repeats
extraction/fine-tuning at higher resolution; use the corresponding feature folder
and reduce `--batch-size` as needed.
Use `--mode full` for full fine-tuning, or `--shared --detectors yolo faster-rcnn`
for a shared encoder/MLP with one output per detector. Frozen regressors can also
be run directly with `train --features <file.npz> --regressor ridge|mlp`.

Models, histories and predictions go to `models/assessors/`; comparison tables,
group-bootstrap MAE intervals and diagnostic plots go to `results/assessors/`.
All PNG figures and CSV tables are saved directly in that folder, without HTML.
Use `--results` to choose another folder; `--output` selects the saved model runs.
`--seeds` and `--fractions` also work in the screening runner. Existing run folders
are not overwritten; use a different `--output` directory for another experiment.
Legacy prediction files lack provenance and trigger an explicit assumption message;
regenerate them from the frozen checkpoints for the actual study. No assessor
training is launched automatically alongside detector fine-tuning.

## Research checks

Use temporary smoke checks when modifying the pipeline and remove them afterward.
No persistent test suite is kept in this research repository.
