#
# To build the extension file mcmc_interf.so use the commands:
#
# Float mode:
#   make purge; python setup_mcmc.py build_ext --inplace; make clean
# Double mode:
#   MCMC_DOUBLE=1 make purge; \
#       MCMC_DOUBLE=1 python setup_mcmc.py build_ext --inplace; make clean
#
# or simply:
#
#   make MCMC_DOUBLE=1
#
# The MCMC_DOUBLE environment variable selects which .pyx is compiled:
#   - not set  -> mcmc_interf_float.pyx  (mcmc_real = float)
#   - set to 1 -> mcmc_interf_double.pyx (mcmc_real = double)
#

# from distutils.core import setup
# from distutils.extension import Extension
# from Cython.Distutils import build_ext

from setuptools import setup, Extension
from Cython.Build import build_ext

import os
import shutil
import numpy as np

bdir = os.getcwd()

# ---------------------------------------------------------------------------
# Locate CUDA on this machine (works for apt-installed and official installer)
# ---------------------------------------------------------------------------
def locate_cuda():
    """
    Returns dict with keys 'include' and 'lib64'.
    Priority:
      1. CUDA_HOME / CUDA_PATH / CUDA_ROOT / CUDAHOME environment variables
      2. Apt package nvidia-cuda-toolkit (/usr/lib/nvidia-cuda-toolkit)
      3. nvcc in PATH (official installer, resolves symlinks)
      4. Common defaults (/usr/local/cuda, /usr/local/cuda-12.2, ...)
    """
    def _paths_from_root(root):
        inc = os.path.join(root, 'include')
        if not os.path.isfile(os.path.join(inc, 'cuda_runtime.h')):
            return None
        lib = os.path.join(root, 'lib64')
        if not os.path.isdir(lib):
            lib = os.path.join(root, 'lib')
        return {'include': inc, 'lib64': lib}

    # 1. Environment variables
    for var in ('CUDA_HOME', 'CUDA_PATH', 'CUDA_ROOT', 'CUDAHOME'):
        if var in os.environ:
            p = _paths_from_root(os.environ[var])
            if p:
                return p

    # 2. Apt package nvidia-cuda-toolkit
    apt_root = '/usr/lib/nvidia-cuda-toolkit'
    if os.path.isfile(os.path.join(apt_root, 'include', 'cuda_runtime.h')):
        # libcudart lives in /usr/lib/x86_64-linux-gnu for apt installs
        return {'include': os.path.join(apt_root, 'include'),
                'lib64':   '/usr/lib/x86_64-linux-gnu'}

    # 3. nvcc in PATH (resolve symlinks)
    nvcc = shutil.which('nvcc')
    if nvcc:
        real_nvcc = os.path.realpath(nvcc)
        root = os.path.dirname(os.path.dirname(real_nvcc))
        p = _paths_from_root(root)
        if p:
            return p

    # 4. Common defaults
    for root in ('/usr/local/cuda', '/usr/local/cuda-12.2',
                 '/usr/local/cuda-12.4', '/usr/local/cuda-13.1'):
        p = _paths_from_root(root)
        if p:
            return p

    raise RuntimeError(
        "CUDA not found. Set CUDA_HOME to the CUDA root "
        "(directory containing include/cuda_runtime.h).")


# ---------------------------------------------------------------------------
# Select .pyx file based on MCMC_DOUBLE
# ---------------------------------------------------------------------------
use_double = bool(os.environ.get('MCMC_DOUBLE'))

if use_double:
    pyx_file = 'mcmc_interf_double.pyx'
    define_macros = [('MCMC_DOUBLE', None)]
    print("Building mcmc_interf in DOUBLE precision (mcmc_interf_double.pyx)")
else:
    pyx_file = 'mcmc_interf_float.pyx'
    define_macros = []
    print("Building mcmc_interf in FLOAT precision (mcmc_interf_float.pyx)")

if not os.path.isfile(pyx_file):
    raise RuntimeError(f"{pyx_file} not found in {bdir}")


# ---------------------------------------------------------------------------
# CUDA paths
# ---------------------------------------------------------------------------
cuda = locate_cuda()
print(f"CUDA include: {cuda['include']}")
print(f"CUDA lib:     {cuda['lib64']}")


# ---------------------------------------------------------------------------
# Build the C library (libmcmcuda.a) via Makefile
# ---------------------------------------------------------------------------
os.system('make')


# ---------------------------------------------------------------------------
# Extension
# ---------------------------------------------------------------------------
ext_mods = Extension("mcmc_interf",
    sources=[pyx_file],
    include_dirs=[np.get_include(), cuda['include']],
    library_dirs=[bdir, cuda['lib64']],
    libraries=['mcmcuda', 'cudart', 'm', 'stdc++'],
    define_macros=define_macros)

setup(
  name='mcmc_interf',
  cmdclass={'build_ext': build_ext},
  ext_modules=[ext_mods])
