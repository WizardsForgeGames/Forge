"""Forge GPU Compute Library -- Python package."""
import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

ROOT = os.path.dirname(os.path.abspath(__file__))
FORGE_ROOT = os.path.dirname(ROOT)  # parent dir with include/ and src/

# Collect all .cu source files that make up the static library.
# We compile them directly into the extension instead of linking
# a separate static lib, which simplifies the build chain.
forge_sources = []
for dirpath, _, filenames in os.walk(os.path.join(FORGE_ROOT, "src")):
    for fn in filenames:
        if fn.endswith((".cu", ".cpp")):
            forge_sources.append(os.path.join(dirpath, fn))

setup(
    name="forge-gpu",
    version="0.1.0",
    description="Forge GPU Compute Library - PyTorch integration",
    author="Darren Sander",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="forge._forge_C",
            sources=[
                os.path.join("forge", "csrc", "bindings.cpp"),
            ] + forge_sources,
            include_dirs=[
                os.path.join(FORGE_ROOT, "include"),
                os.path.join(FORGE_ROOT, "src", "internal"),
            ],
            extra_compile_args={
                "cxx": ["/O2"] if os.name == "nt" else ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "--expt-relaxed-constexpr",
                    "--allow-unsupported-compiler",
                    "-gencode=arch=compute_80,code=sm_80",
                    "-gencode=arch=compute_86,code=sm_86",
                    "-gencode=arch=compute_89,code=sm_89",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
    install_requires=["torch>=2.0"],
)
