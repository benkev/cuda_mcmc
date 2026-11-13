#
# Compile and build mcmc_interf.so Python extension module
# in the srs/ directory. 
#

#TASKDIR = $(PWD)
TASKDIR = $(CURDIR)             #### $(PWD)
MODELDIR = $(TASKDIR)
export MODELDIR       # The current dir must have model.cuh to include

.PHONY:	all clean purge 

all:
	echo "The command CUDA_MCMC is: $(CUDA_MCMC) setup_mcmc.py"
	echo "TASKDIR is: $(TASKDIR)"
	echo "MODELDIR is: $(MODELDIR)"
	cd $(CUDA_MCMC)/src && $(MAKE) purge \
			&& python setup_mcmc.py build_ext --inplace \
			&& $(MAKE) clean && $(MAKE) install && cd $(TASKDIR)
	@echo "  "
	@echo MODEL is included from $(MODELDIR)/model.cuh
	@echo "  "

clean:
	cd $(CUDA_MCMC)/src && $(MAKE) clean && cd $(TASKDIR)

purge:
	cd $(CUDA_MCMC)/src && $(MAKE) purge && cd $(TASKDIR)

