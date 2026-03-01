import os
import sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Platform-specific compiler flags
if sys.platform == 'win32':
    nvcc_flags = ["-allow-unsupported-compiler", "-Xcompiler", "/Zc:preprocessor"]
    cxx_flags = ["/Zc:preprocessor"]
else:
    nvcc_flags = []
    cxx_flags = []

setup(
    name="attention_cuda",
    ext_modules=[
        CUDAExtension(
            "attention_cuda",
            [
                "csrc/attention_ext.cpp",
                "csrc/flash_double_backward_v7.cu",
            ],
            libraries=["cublas"],
            extra_compile_args={
                "nvcc": nvcc_flags,
                "cxx": cxx_flags,
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
