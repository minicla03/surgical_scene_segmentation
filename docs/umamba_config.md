# U-Mamba Setup Guide

This document summarizes the setup used to run **U-Mamba** alongside the main `surgical_scene_segmentation` project.
Because U-Mamba requires an older Python/PyTorch stack than the main project, it is installed in a **separate Conda environment**.

https://github.com/bowang-lab/U-Mamba
---

## 1. Create the Conda environment

U-Mamba requires Python 3.10.

```bash
conda create -n umamba python=3.10 -y
conda activate umamba
```

---

## 2. Install PyTorch

The cluster GPUs are **NVIDIA Tesla P100** GPUs:

- Architecture: Pascal
- Compute Capability: 6.0
- CUDA architecture: `sm_60`

The U-Mamba environment uses **PyTorch 2.0.1 with CUDA 11.8**.

```bash
pip install torch==2.0.1 torchvision==0.15.2 \
    --index-url https://download.pytorch.org/whl/cu118
```

Additional build dependencies:

```bash
pip install packaging ninja
```

The relevant environment configuration can be find into the repo: docs/env_umamba.yaml

---

## 3. CUDA architecture issue on Tesla P100

The standard precompiled wheels of `causal-conv1d` and `mamba-ssm` do not provide kernels for the Tesla P100's `sm_60` architecture.

As a consequence, using the standard wheels can result in an error such as:

```text
no kernel image is available for execution on the device
```

Setting `TORCH_CUDA_ARCH_LIST` alone is not sufficient in this case because the packages' `setup.py` files explicitly define their CUDA architectures.

Therefore, both packages must be **compiled from source with explicit `sm_60` support**.

---

## 4. Install `causal-conv1d` from source

First remove any previously installed version:

```bash
pip uninstall -y causal-conv1d
```

Install a compatible NumPy version:

```bash
pip install "numpy==1.26.4"
```

Clone the required version:

```bash
git clone --branch v1.2.0.post2 \
    https://github.com/Dao-AILab/causal-conv1d.git

cd causal-conv1d
```

Open `setup.py` and locate:

```python
cc_flag = []
```

Add the following immediately after it:

```python
cc_flag += ["-gencode", "arch=compute_60,code=sm_60"]
```

This explicitly adds Tesla P100 support.

Then build and install from source:

```bash
CAUSAL_CONV1D_FORCE_BUILD=TRUE \
pip install . --no-build-isolation --no-cache-dir
```

---

## 5. Install `mamba-ssm` from source

Clone the required version:

```bash
git clone --branch v1.2.0.post1 \
    https://github.com/state-spaces/mamba.git

cd mamba
```

Again, open `setup.py` and locate:

```python
cc_flag = []
```

Add:

```python
cc_flag += ["-gencode", "arch=compute_60,code=sm_60"]
```

Then build from source:

```bash
MAMBA_FORCE_BUILD=TRUE \
pip install . --no-build-isolation --no-cache-dir
```

---

## 6. Verify the installation

Check that the CUDA extension was compiled correctly:

```bash
python -c "import causal_conv1d_cuda as m; print(m.__file__)"
```

The resulting path can then be inspected with:

```bash
cuobjdump --list-elf \
$(python -c "import causal_conv1d_cuda as m; print(m.__file__)") \
| grep sm_
```

The output should contain:

```text
sm_60
```

This confirms that the CUDA extension contains a kernel compiled for the Tesla P100 architecture.

---

## 7. Basic Mamba test

After installing the dependencies, test whether Mamba can be imported and moved to the GPU:

```python
import torch
from mamba_ssm import Mamba

model = Mamba(d_model=16)
model = model.cuda()

x = torch.randn(2, 32, 16).cuda()

y = model(x)

print(y.shape)
```

If this runs successfully, the Mamba CUDA extension is working on the P100.

---

## 8. Precision considerations

Tesla P100 GPUs do **not support BF16**.

Therefore, U-Mamba should be run using:

- FP32, or
- FP16 where supported.

BF16 should not be enabled.

FP16 can be used, but the P100 does not have Tensor Cores, so the performance benefit is smaller than on more recent NVIDIA architectures.

---

## 9. U-Mamba and the main project

The U-Mamba repository is kept separate from the main thesis project because it requires a different software environment.

The structure is:

```text
~/tesi/
├── surgical_scene_segmentation/
│   └── main thesis environment
│
└── U-Mamba/
    └── umamba/
        └── nnunetv2/
            └── nets/
                └── UMambaBot_2d.py
```

The two environments are therefore independent:

```text
surgical_scene_segmentation
        │
        └── main Python environment

U-Mamba
        │
        └── umamba Python 3.10 environment
```

The main project can invoke the U-Mamba implementation without requiring U-Mamba's dependencies to be installed into the main environment.

---


## Summary

The final setup is:

| Component | Version / Configuration |
|---|---|
| Python | 3.10 |
| PyTorch | 2.0.1 |
| CUDA | 11.8 |
| GPU | NVIDIA Tesla P100 |
| Compute Capability | 6.0 (`sm_60`) |
| causal-conv1d | 1.2.0.post2, compiled from source |
| mamba-ssm | 1.2.0.post1, compiled from source |
| NumPy | 1.26.4 |
| BF16 | Not supported |
| Environment | Separate `umamba` Conda environment |

The key modification required for the cluster hardware is the explicit addition of:

```python
cc_flag += ["-gencode", "arch=compute_60,code=sm_60"]
```

to the CUDA build configuration of both `causal-conv1d` and `mamba-ssm`.