# 
# To build the extension file mcmc_interf.so use the commands:
#
# $ make purge; python setup_mcmc.py build_ext --inplace;  make clean
#

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import os
import numpy as np

bdir = os.getcwd()
os.system('make')

ext_mods = Extension("mcmc_interf",
    sources = ["mcmc_interf.pyx"],
    include_dirs = [np.get_include(),
      '/usr/local/cuda-13.1/targets/x86_64-linux/include'],  # ADDED
#    library_dirs = [bdir, '/usr/lib/x86_64-linux-gnu'],
    library_dirs = [bdir, '/usr/local/cuda-12.2/targets/x86_64-linux/lib'],
    libraries = ['mcmcuda', 'cudart', 'm', 'stdc++'])

setup(
  name = 'mcmc_interf',
  cmdclass = {'build_ext': build_ext},
  ext_modules = [ext_mods]
)
