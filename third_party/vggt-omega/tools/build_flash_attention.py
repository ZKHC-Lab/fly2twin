"""Build upstream FA 1.0.9 as an isolated Torch extension, not a global package.

Sources and CUTLASS are arguments; the build script lives outside downloaded
sources. Use python -I. Mirrors upstream setup.py flags but compiles sm_87 only,
with one ninja job and one NVCC thread for bounded host memory.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch
from torch.utils.cpp_extension import load

p = argparse.ArgumentParser()
p.add_argument('--source', type=Path, required=True)
p.add_argument('--cutlass', type=Path, required=True)
p.add_argument('--build', type=Path, required=True)
a = p.parse_args()
base = a.source.resolve() / 'csrc/flash_attn'
sources = [base / x for x in (
    'fmha_api.cpp', 'src/fmha_fwd_hdim32.cu', 'src/fmha_fwd_hdim64.cu', 'src/fmha_fwd_hdim128.cu',
    'src/fmha_bwd_hdim32.cu', 'src/fmha_bwd_hdim64.cu', 'src/fmha_bwd_hdim128.cu',
    'src/fmha_block_fprop_fp16_kernel.sm80.cu', 'src/fmha_block_dgrad_fp16_kernel_loop.sm80.cu')]
a.build.mkdir(parents=True, exist_ok=True)
os.environ['MAX_JOBS'] = '1'
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.7'
compat = ['-DOLD_GENERATOR_PATH'] if (Path(torch.__file__).parent / 'include/ATen/CUDAGeneratorImpl.h').exists() else []
t = time.monotonic()
module = load(name='drone3d_flash_attn_v109', sources=[str(x) for x in sources],
    extra_include_paths=[str(base), str(base / 'src'), str(a.cutlass.resolve() / 'include')],
    extra_cflags=['-O3', '-std=c++17'] + compat,
    extra_cuda_cflags=['-O3', '-std=c++17', '-U__CUDA_NO_HALF_OPERATORS__',
        '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__',
        '-U__CUDA_NO_BFLOAT16_CONVERSIONS__', '--expt-relaxed-constexpr',
        '--expt-extended-lambda', '--use_fast_math', '--threads', '1'] + compat,
    build_directory=str(a.build.resolve()), verbose=True)
result = dict(seconds=time.monotonic()-t, module=module.__file__, torch=torch.__version__, cuda=torch.version.cuda,
    abi=torch._C._GLIBCXX_USE_CXX11_ABI, capability=torch.cuda.get_device_capability(0),
    sources_sha256={str(x.relative_to(base)): hashlib.sha256(x.read_bytes()).hexdigest() for x in sources})
(a.build / 'build_result.json').write_text(json.dumps(result, indent=2))
print('BUILD_COMPLETED', json.dumps(result), flush=True)
