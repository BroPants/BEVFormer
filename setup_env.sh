#!/usr/bin/env bash
# BEVFormer environment setup — adapted for RTX 4090 (sm_89) + CUDA 12.x driver
#
# Deviations from docs/install.md:
#   - PyTorch 1.13.1+cu118  (1.9.1+cu111 does not support sm_89 / Ada Lovelace)
#   - mmcv-full 1.7.2        (last 1.x release with torch1.13+cu118 prebuilt wheel)
#   - mmdet 2.28.2 / mmseg 0.30.0  (compatible 2.x releases)
#   - numpy 1.23.5 / numba 0.56.4  (numpy 1.19.5 conflicts with Python 3.8 + newer tools)
#   mmdet3d 0.17.1 is unchanged — installed from source using conda CUDA 11.8 toolkit

set -e
CONDA_ENV=bevformer
MMDET3D_DIR="$HOME/mmdetection3d"

echo "=== Step 1: Create conda environment ==="
conda create -n "$CONDA_ENV" python=3.8 -y

echo ""
echo "=== Step 2: Install CUDA 11.8 toolkit (needed by nvcc to build mmdet3d ops) ==="
# cudatoolkit-dev was removed from conda-forge; use NVIDIA's official channel instead.
# cuda-toolkit from nvidia/label/cuda-11.8.0 includes nvcc + full dev headers.
conda run -n "$CONDA_ENV" conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit -y

echo ""
echo "=== Step 3: Install PyTorch 1.13.1 + cu117 ==="
# torch 1.13.x was only released for cu116/cu117; cu118 builds start at torch 2.0.
# CUDA 11.8 toolkit (installed in Step 2) can compile extensions for cu117 torch — ABI compatible.
conda run -n "$CONDA_ENV" pip install \
  torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 \
  -f https://download.pytorch.org/whl/torch_stable.html

echo ""
echo "=== Step 4: Install mmcv-full (prebuilt wheel, no compilation needed) ==="
conda run -n "$CONDA_ENV" pip install mmcv-full==1.7.2 \
  -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html

echo ""
echo "=== Step 5: Install mmdet and mmsegmentation ==="
conda run -n "$CONDA_ENV" pip install mmdet==2.28.2 mmsegmentation==0.30.0

echo ""
echo "=== Step 6: Clone and install mmdet3d v0.17.1 from source ==="
if [ ! -d "$MMDET3D_DIR" ]; then
  git clone https://github.com/open-mmlab/mmdetection3d.git "$MMDET3D_DIR"
fi
cd "$MMDET3D_DIR"
# Only checkout if not already on v0.17.1 (avoids clobbering THC patches applied above)
CURRENT_TAG=$(git describe --tags --exact-match 2>/dev/null || echo "")
if [ "$CURRENT_TAG" != "v0.17.1" ]; then
  git checkout v0.17.1
fi
# Patch: remove THC/THC.h (removed in PyTorch 1.13) from the 6 ops that still include it
for cpp in \
  mmdet3d/ops/ball_query/src/ball_query.cpp \
  mmdet3d/ops/knn/src/knn.cpp \
  mmdet3d/ops/furthest_point_sample/src/furthest_point_sample.cpp \
  mmdet3d/ops/group_points/src/group_points.cpp \
  mmdet3d/ops/gather_points/src/gather_points.cpp \
  mmdet3d/ops/interpolate/src/interpolate.cpp; do
  sed -i '/#include <THC\/THC.h>/d' "$cpp"
  sed -i '/extern THCState \*state;/d' "$cpp"
done
echo "  THC patch applied."
# Patch 2: add missing ATen/cuda/CUDAContext.h (was transitively included via THC/THC.h)
for cpp in \
  mmdet3d/ops/ball_query/src/ball_query.cpp \
  mmdet3d/ops/group_points/src/group_points.cpp \
  mmdet3d/ops/interpolate/src/interpolate.cpp; do
  if ! grep -q "ATen/cuda/CUDAContext.h" "$cpp"; then
    sed -i '1s|^|#include <ATen/cuda/CUDAContext.h>\n|' "$cpp"
  fi
done
echo "  CUDAContext.h patch applied."
# Patch 3: relax mmcv upper version bound (0.17.1 requires <=1.4.0, but 1.x is API-compatible)
sed -i "s/mmcv_maximum_version = '1.4.0'/mmcv_maximum_version = '1.8.0'  # relaxed/" \
  mmdet3d/__init__.py
echo "  mmcv version bound relaxed."
# Patch 4: mmcv 1.7.x registers its own SparseConv2d; mmdet3d 0.17.1 must force-override
sed -i 's/@CONV_LAYERS\.register_module()/@CONV_LAYERS.register_module(force=True)/g' \
  mmdet3d/ops/spconv/conv.py
echo "  spconv force-register patch applied."
# Patch 5: numba.errors moved to numba.core.errors in numba >= 0.53
sed -i 's/from numba\.errors import/from numba.core.errors import/' \
  mmdet3d/datasets/pipelines/data_augment_utils.py
echo "  numba.errors patch applied."
# Install ninja for faster compilation, then build
conda run -n "$CONDA_ENV" pip install ninja -q
CONDA_PREFIX="$(conda info --base)/envs/$CONDA_ENV"
CUDA_HOME="$CONDA_PREFIX" conda run -n "$CONDA_ENV" python setup.py develop
cd -

echo ""
echo "=== Step 7: Install remaining Python dependencies ==="
conda run -n "$CONDA_ENV" pip install \
  einops fvcore seaborn iopath==0.1.9 timm==0.6.13 \
  typing-extensions==4.5.0 pylint "ipython==8.12" \
  "numpy==1.23.5" "matplotlib==3.5.2" "numba==0.56.4" \
  "pandas==1.4.4" "scikit-image==0.19.3" "setuptools==59.5.0"

echo ""
echo "=== Step 8: Install Detectron2 ==="
conda run -n "$CONDA_ENV" python -m pip install \
  'git+https://github.com/facebookresearch/detectron2.git'

echo ""
echo "=== Step 9: Download pretrained backbone ==="
mkdir -p /home/jiaqi/Desktop/code/BEVFormer/ckpts
CKPT=/home/jiaqi/Desktop/code/BEVFormer/ckpts/r101_dcn_fcos3d_pretrain.pth
if [ ! -f "$CKPT" ]; then
  wget -O "$CKPT" \
    https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth
else
  echo "Checkpoint already exists, skipping."
fi

echo ""
echo "=== Done! Activate with: conda activate $CONDA_ENV ==="
echo "=== Quick smoke test: ==="
echo "    conda activate $CONDA_ENV"
echo "    python -c \"import torch; print(torch.__version__, torch.cuda.get_device_name(0))\""
