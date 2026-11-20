# Installation and resource separation

2026-10-26

The package pins public gsplat 1.5.3. Its own strategy callback preserves the reference schedule: no opacity reset at step zero, then resets at positive multiples of `reset_every` before `refine_stop_iter`. Public 1.5.2 resets at zero; public 1.5.3 contains `== 0 & step > 0`, a chained comparison that never resets. The explicit callback avoids both differences and is regression-tested at steps 0, 3000, 6000 and 15000. A historical checkout's version string alone is insufficient to identify this behavior.

## CUDA toolkit

Install PyTorch/torchvision for the intended CUDA version, a matching CUDA compiler toolkit, a C++17 compiler, ninja, and fused-ssim. A CUDA-capable PyTorch wheel alone does not provide `nvcc` and headers required for gsplat JIT compilation.

The tested machine used Python 3.10.20, PyTorch 2.7.1+cu128, CUDA toolkit 12.8 and architecture 12.0. These are tested settings, not universal hardware defaults. See [observed dependencies](../../requirements-lock/tested-environment.txt); this is not a complete offline binary lockfile.

For a conda toolkit whose headers live under `targets/x86_64-linux`, the environment must point `CUDA_HOME` to the **environment root**, not the target directory. Calling the target-directory nvcc symlink can break its relative lookup of the `cicc` compiler. Example, after activating the intended environment:

```bash
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
export CPLUS_INCLUDE_PATH="$CONDA_PREFIX/targets/x86_64-linux/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export C_INCLUDE_PATH="$CONDA_PREFIX/targets/x86_64-linux/include${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
export LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
nvcc --version
```

For a conventional system CUDA installation, use that toolkit's root instead. Do not mutate package modules to configure the host toolchain.

## Compile outside the GPU lock

Explicitly choose the GPU architecture; the example below is for SM 12.0 only. Disabling device visibility ensures the prebuild does not consume a GPU. Compilation is CPU/RAM work and must not block other users' GPU experiments.

```bash
export TORCH_EXTENSIONS_DIR="$PWD/.local/torch_extensions"
mkdir -p "$TORCH_EXTENSIONS_DIR"
CUDA_VISIBLE_DEVICES='' TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=2 \
  python -c 'from gsplat.cuda._backend import _C; assert _C is not None'
```

On memory-constrained shared machines, wrap compilation with a memory limit (the local acceptance used a 16 GB systemd user scope). If JIT was interrupted after generating the ninja build, resume `ninja -C <extension-build-directory> -j 2` outside the GPU lock.

## XFeat (bundle adjustment only)

`pose.refine: ba` matches features with [XFeat](https://github.com/verlab/accelerated_features) in addition to SIFT. XFeat is not a pip dependency: clone it, put the checkout on `PYTHONPATH` so that `modules.xfeat` imports, and point `pose.xfeat_weights` at its `weights/xfeat.pt`. Nothing is downloaded at run time; a missing module or checkpoint is an error.

```bash
git clone https://github.com/verlab/accelerated_features.git
export PYTHONPATH="$PWD/accelerated_features${PYTHONPATH:+:$PYTHONPATH}"
```

## Separate CPU and GPU phases

External poses (`pose.provider: external`, `pose.refine: none`) with cached depth support CPU-only preparation and fusion, including the phone B0 protocol:

```bash
recon run --config my_scene.yaml --through fuse
```

Then use a nonblocking shared lock **only for actual GPU phases**, after higher-priority jobs have finished:

```bash
flock -n "$GPU_LOCK" recon run --config my_scene.yaml --only train
flock -n "$GPU_LOCK" recon run --config my_scene.yaml --only eval
recon run --config my_scene.yaml --only export
```

A busy nonblocking lock is not a training failure: do other CPU work and retry later, not a tight polling loop. New-image prior pose estimation includes XFeat GPU matching; fresh Omega inference also needs GPU access. The first release's combined `pose` stage does not expose separate SIFT/BA/XFeat CLI substages; schedule that fresh-input stage deliberately rather than assuming it is entirely CPU-only.

Before evaluation, explicitly install/download the licensed LPIPS/AlexNet weights into the standard torch cache. The evaluator fails rather than silently downloading them. Test-RGB camera fitting is evaluation-only and its results must be reported separately from fixed-camera scores.
