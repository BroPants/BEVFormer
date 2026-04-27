"""Quick environment verification: loads config, builds model+dataset, runs one forward pass."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings('ignore')

import torch
from mmcv import Config
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader

CONFIG = 'projects/configs/bevformer/bevformer_tiny.py'
CHECKPOINT = 'ckpts/bevformer_tiny_epoch_24.pth'
BACKBONE_ONLY = 'ckpts/r101_dcn_fcos3d_pretrain.pth'

def main():
    print("=== BEVFormer environment verification ===\n")

    # 1. torch + cuda
    print(f"[1] PyTorch {torch.__version__} | CUDA {torch.version.cuda}")
    assert torch.cuda.is_available(), "CUDA not available!"
    print(f"    GPU: {torch.cuda.get_device_name(0)}\n")

    # 2. Plugin import
    print("[2] Loading plugin modules...")
    import projects.mmdet3d_plugin  # noqa: registers custom modules
    print("    OK\n")

    # 3. Config
    print(f"[3] Parsing config: {CONFIG}")
    cfg = Config.fromfile(CONFIG)
    cfg.data.test.test_mode = True
    print("    OK\n")

    # 4. Dataset
    print("[4] Building validation dataset...")
    dataset = build_dataset(cfg.data.test)
    print(f"    {len(dataset)} samples found")
    loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=0,
        dist=False, shuffle=False)
    batch = next(iter(loader))
    # mmcv wraps tensors in DataContainer; unwrap for inspection
    from mmcv.parallel import DataContainer
    def unwrap(x):
        return x.data if isinstance(x, DataContainer) else x
    imgs_raw = batch['img']
    if isinstance(imgs_raw, list):
        imgs_raw = [unwrap(x) for x in imgs_raw]
        shape_info = [x[0].shape if isinstance(x, list) else x.shape for x in imgs_raw]
    else:
        shape_info = unwrap(imgs_raw).shape
    print(f"    Batch img shape: {shape_info}\n")

    # 5. Model init
    print("[5] Building model...")
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    model = model.cuda().eval()

    # Load full checkpoint if available, else just the backbone
    if os.path.exists(CHECKPOINT):
        print(f"    Loading full checkpoint: {CHECKPOINT}")
        from mmcv.runner import load_checkpoint
        load_checkpoint(model, CHECKPOINT, map_location='cpu', strict=False)
    else:
        print(f"    Full checkpoint not ready yet — using random weights for pipeline check")
        print(f"    (backbone pretrained: {BACKBONE_ONLY})")

    # 6. Forward pass via MMDataParallel (handles DataContainer unwrapping)
    print("\n[6] Running forward pass (2 frames to exercise temporal BEV)...")
    from mmcv.parallel import MMDataParallel
    from mmdet3d.apis import single_gpu_test
    model = MMDataParallel(model, device_ids=[0])
    # Run on 2 batches only (enough to verify temporal pipeline)
    results = []
    for i, data in enumerate(loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.extend(result)
        if i >= 1:
            break

    r = results[-1]
    if 'boxes_3d' in r:
        boxes = r['boxes_3d']
        scores = r.get('scores_3d', None)
        print(f"    Detected {len(boxes)} boxes in last sample")
        if scores is not None and len(scores) > 0:
            print(f"    Top score: {float(scores.max()):.3f}")
    else:
        print(f"    Result keys: {list(r.keys())}")

    print("\n=== All checks passed — environment is correctly configured ===")

if __name__ == '__main__':
    main()
