#
# template/Makefile
#
# Application-level Makefile. It calls src/Makefile to build libmcmcuda.a
# with model.cuh from the current (application) directory, then builds
# the Python extension module mcmc_interf.*.so via setup_mcmc.py.
#
# Usage:
#   make                 -> float  mode
#   make MCMC_DOUBLE=1   -> double mode
#
# The CUDA_MCMC environment variable must point to the cuda_mcmc root
# (the directory containing src/ and template/).
#

TASKDIR  = $(CURDIR)
MODELDIR = $(TASKDIR)
export MODELDIR

.PHONY: all clean purge

all:
	cd $(CUDA_MCMC)/src && $(MAKE) purge \
		&& $(MAKE) $(if $(MCMC_DOUBLE),MCMC_DOUBLE=$(MCMC_DOUBLE)) \
		&& python setup_mcmc.py build_ext --inplace \
		&& $(MAKE) clean && $(MAKE) install \
		&& cd $(TASKDIR)
	@echo "  "
	@echo "MODEL is included from $(MODELDIR)/model.cuh"
	@echo "  "

clean:
	cd $(CUDA_MCMC)/src && $(MAKE) clean && cd $(TASKDIR)

purge:
	cd $(CUDA_MCMC)/src && $(MAKE) purge && cd $(TASKDIR)