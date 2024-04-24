import glob
import os

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
include_dirs = [os.path.join(ROOT_DIR, "include")]

sources = glob.glob('*.cpp') + glob.glob('*.cu')

setup(
    name='cppcuda_bn',
    version='1.0',
    author='kasawa',
    ext_modules=[
        CUDAExtension(
            name='cppcuda_bn',
            sources=sources,
            include_dirs=include_dirs
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
