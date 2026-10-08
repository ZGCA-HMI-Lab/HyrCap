"""Build the native temporal attention extension."""

from glob import glob
from pathlib import Path

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME


def get_extension():
    source_root = Path(__file__).resolve().parent / "src"
    if not torch.cuda.is_available() or CUDA_HOME is None:
        raise RuntimeError("A compatible accelerator toolchain is required.")
    sources = [
        *glob(str(source_root / "*.cpp")),
        *glob(str(source_root / "cuda" / "*.cu")),
    ]
    return CUDAExtension(
        "MultiScaleTemporalDeformableAttention",
        sources,
        include_dirs=[str(source_root)],
        define_macros=[("WITH_CUDA", None)],
        extra_compile_args={
            "cxx": [],
            "nvcc": [
                "-DCUDA_HAS_FP16=1",
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
            ],
        },
    )


setup(
    name="MultiScaleTemporalDeformableAttention",
    version="1.0",
    ext_modules=[get_extension()],
    cmdclass={"build_ext": BuildExtension},
)
