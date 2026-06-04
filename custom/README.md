# Custom Notes

For general usage, training, and deployment, see the upstream [README](../README.md) and [projects/rtmpose/README.md](../projects/rtmpose/README.md).

This directory contains setup notes and patches specific to this fork.

---

## uv Environment Setup

Uses Python 3.12, CUDA 12.1, torch 2.4.0. Pinned deps: [requirements-lock.txt](../requirements-lock.txt).

```bash
uv venv .venv --python 3.12

# PyTorch (CUDA 12.1)
uv pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121 --python .venv/bin/python

# mmcv prebuilt wheel for CUDA 12.1 / torch 2.4
uv pip install mmcv==2.2.0 \
    --find-links https://download.openmmlab.com/mmcv/dist/cu121/torch2.4.0/index.html \
    --python .venv/bin/python

# mmengine + mmdet
uv pip install "mmengine>=0.4.0,<1.0.0" "mmdet>=3.0.0" --python .venv/bin/python

# System headers needed to compile xtcocotools
sudo apt-get install -y python3.12-dev

# Runtime deps (--no-build-isolation so xtcocotools can find numpy)
uv pip install numpy --python .venv/bin/python
uv pip install opencv-python pillow scipy matplotlib json_tricks munkres \
    xtcocotools requests --python .venv/bin/python --no-build-isolation

# setuptools<72 required — newer versions remove pkg_resources, breaking mmengine
uv pip install "setuptools<72" --python .venv/bin/python

# mmpose editable install
uv pip install -e . --no-build-isolation --python .venv/bin/python

# ONNX export and standalone inference
uv pip install onnx onnxruntime --python .venv/bin/python

# Apply patches to installed wheels
.venv/bin/python custom/patches/apply_patches.py

# Create .mim symlinks for mmengine config resolution
mkdir -p mmpose/.mim
for f in tools configs demo model-index.yml dataset-index.yml; do
    [ -e "$f" ] && ln -sfn "$(realpath $f)" mmpose/.mim/$f
done
mkdir -p .venv/lib/python3.12/site-packages/mmpose
ln -sfn "$(realpath mmpose/.mim)" .venv/lib/python3.12/site-packages/mmpose/.mim
```

---

## Patches

### 1. mmdet mmcv version cap (`custom/patches/apply_patches.py`)

`mmdet==3.3.0` hardcodes `mmcv_maximum_version = '2.2.0'` (exclusive), rejecting `mmcv==2.2.0`.
The patch script bumps the cap to `2.3.0` in the installed wheel. **Re-run after recreating the venv.**

### 2. `draw_circles` radius type error

`mmpose/visualization/opencv_backend_visualizer.py`: `int(radius)` raises `TypeError` in NumPy 2.x
when `radius` is a 0-d array. Fixed with `int(np.asarray(radius).item())`. Tracked in git.
