# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BEVFormer is a transformer-based multi-camera 3D object detection framework that generates Bird's-Eye-View (BEV) representations from multi-view images. It builds on top of MMDetection3D and supports both BEVFormer v1 (ECCV 2022) and BEVFormerV2 variants.

## Environment Setup

Requires a specific software stack — versions matter:
- Python 3.8, PyTorch 1.9.1+cu111
- mmcv-full==1.4.0, mmdet==2.14.0, mmsegmentation==0.14.1
- mmdet3d==0.17.1 (installed from source)
- Additional: `pip install einops fvcore seaborn timm==0.6.13 detectron2`

See `docs/install.md` for the complete step-by-step instructions.

## Key Commands

**Dataset preparation** (nuScenes, after downloading raw data to `./data/nuscenes/`):
```bash
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes \
  --extra-tag nuscenes --version v1.0 --canbus ./data
```

**Distributed training** (8 GPUs):
```bash
./tools/dist_train.sh ./projects/configs/bevformer/bevformer_base.py 8
```

**FP16 training**:
```bash
./tools/fp16/dist_train.sh ./projects/configs/bevformer_fp16/bevformer_tiny_fp16.py 8
```

**Distributed evaluation** (8 GPUs):
```bash
./tools/dist_test.sh ./projects/configs/bevformer/bevformer_base.py /path/to/ckpt.pth 8 \
  --eval bbox
```

**Single-GPU evaluation** (gives better NDS due to temporal continuity across frames):
```bash
python tools/test.py ./projects/configs/bevformer/bevformer_base.py /path/to/ckpt.pth \
  --eval bbox
```

> Note: Single-GPU evaluation typically yields ~1% higher NDS than 8-GPU evaluation because multi-GPU evaluation breaks the temporal sequence continuity.

## mmdet3d 0.17.1 Compatibility Patches

When installing mmdet3d v0.17.1 with PyTorch 1.13 + mmcv-full 1.7.x, five source patches are required (all applied automatically by `setup_env.sh`):

| # | File | Problem | Fix |
|---|---|---|---|
| 1 | 6 ops `*.cpp` files | `THC/THC.h` removed in PyTorch 1.13 | Remove the `#include` line |
| 2 | `ball_query.cpp`, `group_points.cpp`, `interpolate.cpp` | Lost `at::cuda::getCurrentCUDAStream` after THC removed | Add `#include <ATen/cuda/CUDAContext.h>` |
| 3 | `mmdet3d/__init__.py` | Hard-coded `mmcv <= 1.4.0` | Change upper bound to `1.8.0` |
| 4 | `ops/spconv/conv.py` | mmcv 1.7 registers its own `SparseConv2d`; mmdet3d's duplicate registration fails | Add `force=True` to all 10 `@CONV_LAYERS.register_module()` decorators |
| 5 | `datasets/pipelines/data_augment_utils.py` | `numba.errors` moved to `numba.core.errors` in numba ≥ 0.53 | Update import path |

## Environment Verification

After setup, run:
```bash
conda activate bevformer
cd /path/to/BEVFormer
python tools/verify_env.py
```
Expected output ends with `All checks passed — environment is correctly configured`.

## Architecture

The plugin code lives entirely in `projects/mmdet3d_plugin/` — MMDetection3D's plugin system loads it automatically when training/testing.

### Core Data Flow

1. **Input**: 6 surround-view camera images per timestep + CAN bus ego-pose data
2. **Backbone**: ResNet extracts multi-scale image features (FPN neck)
3. **BEV Encoder** (`modules/encoder.py`): Iterates BEVFormerEncoder layers, each containing:
   - **Temporal Self-Attention** (`temporal_self_attention.py`): Fuses current BEV queries with BEV features from the previous timestep (warped using ego-motion)
   - **Spatial Cross-Attention** (`spatial_cross_attention.py`): Projects each BEV query to 3D reference points, lifts them onto multi-view image features via deformable attention
4. **BEV Decoder** (`modules/decoder.py`): Standard transformer decoder with 900 object queries attending to the BEV feature map
5. **BEVFormerHead** (`dense_heads/bevformer_head.py`): Predicts 3D bounding boxes; uses Hungarian matching and focal/L1 losses

### Key Design Points

- **Temporal queue**: Configured as `queue_length` in dataset config (e.g., 4 for base). The detector maintains a rolling BEV feature history accessed via `self.prev_bev`.
- **BEV grid**: `bev_h × bev_w` learnable queries (200×200 for base, 50×50 for tiny), covering `point_cloud_range` in the XY plane.
- **Deformable attention**: Custom CUDA kernel in `multi_scale_deformable_attn_function.py`; the `MultiScaleDeformableAttention` module from mmcv is reused for the decoder cross-attention.
- **NMS-free detection**: Post-processing uses `NMSFreeCoder` — no NMS is applied; the Hungarian assignment during training handles duplicate suppression.

### Detector Variants

| Class | File | Use case |
|---|---|---|
| `BEVFormer` | `detectors/bevformer.py` | Standard v1 training |
| `BEVFormer_fp16` | `detectors/bevformer_fp16.py` | Mixed-precision v1 |
| `BEVFormerV2` | `detectors/bevformerV2.py` | V2 with DD3D components |

### Config Hierarchy

```
projects/configs/
├── _base_/               # Shared base configs (runtime, dataset, schedules)
├── bevformer/            # V1 configs: tiny / small / base
├── bevformer_fp16/       # FP16 variants
├── bevformerv2/          # V2 configs: r50-t1/t2/t8, 24ep/48ep variants
└── datasets/             # Standalone dataset configs
```

Configs follow mmdet's inheritance (`_base_ = [...]`). Model capacity is controlled by `embed_dims`, `num_layers`, `bev_h`, `bev_w`, and `queue_length`.

### Performance Reference (nuScenes val)

| Config | NDS | mAP | GPU Memory |
|---|---|---|---|
| bevformer_tiny | 35.4% | 25.2% | ~6.5GB |
| bevformer_small | 47.9% | 37.0% | ~8.5GB |
| bevformer_base | 51.7% | 41.6% | ~28.5GB |
| bevformerv2-r50-t2-24ep | 52.6% | 42.6% | — |
| bevformerv2-r50-t8-24ep | 55.3% | 45.0% | — |

## Data Layout

Expected directory structure under the project root:
```
data/
└── nuscenes/
    ├── maps/
    ├── samples/
    ├── sweeps/
    ├── v1.0-trainval/
    ├── can_bus/               # CAN bus expansion data
    ├── nuscenes_infos_temporal_train.pkl
    └── nuscenes_infos_temporal_val.pkl
```

Pretrained backbone checkpoint (required before training base/small):
```
ckpts/r101_dcn_fcos3d_pretrain.pth
```
