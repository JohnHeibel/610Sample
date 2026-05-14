import os
import sys
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

if sys.platform == 'win32':
    nvcc_flags = ["-allow-unsupported-compiler", "-Xcompiler", "/Zc:preprocessor", "-lineinfo"]
    cxx_flags = ["/Zc:preprocessor"]
else:
    nvcc_flags = ["-lineinfo"]
    cxx_flags = []

repo_root = os.path.dirname(os.path.abspath(__file__))
cutlass_include = os.path.join(repo_root, "third_party", "cutlass", "include")
cutlass_util_include = os.path.join(repo_root, "third_party", "cutlass", "tools", "util", "include")

include_dirs = []
if os.path.isdir(cutlass_include):
    include_dirs.append(cutlass_include)
if os.path.isdir(cutlass_util_include):
    include_dirs.append(cutlass_util_include)

setup(
    name="flash_v9",
    packages=find_packages(exclude=("tests", "bench", "profiling", "paper", "third_party")),
    ext_modules=[
        CUDAExtension(
            "flash_v9_cuda",
            sources=[
                "csrc/attention_ext.cpp",
                "csrc/flash_v9/forward.cu",
                "csrc/flash_v9/backward.cu",
                "csrc/flash_v9/double_backward.cu",
            ],
            include_dirs=include_dirs,
            libraries=["cublas"],
            extra_compile_args={
                "nvcc": nvcc_flags,
                "cxx": cxx_flags,
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
