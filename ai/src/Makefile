#
# src/Makefile
#
# Builds libmcmcuda.a (mcmcjob.o + gpu_mcmc.o) for the CUDA portion of
# cuda_mcmc, and some utility programs. The library is used by
# setup_mcmc.py to build the Python extension module mcmc_interf.*.so.
#
# Two precision modes:
#   make                 -> float  (mcmc_real = float)
#   make MCMC_DOUBLE=1   -> double (mcmc_real = double)
#
# The MODELDIR environment variable must point to the application
# directory containing model.cuh. It is set by the application-level
# Makefile (template/Makefile or a copy of it).
#
# GPU architectures: sm_61 (GTX 1050 Ti), sm_86 (A10), sm_89 (L4).
# If you have a different GPU, adjust the -gencode lines below.
#

CC      = gcc
CXX     = nvcc
CXXFLAGS = -g -Xcompiler -fPIC -Xcompiler -fno-exceptions
OBJECTS  = mcmcjob.o gpu_mcmc.o
BASEDIR  = $(CURDIR)

# --- GPU architectures -----------------------------------------------------
# Add or remove -gencode lines for your GPUs.
# sm_61 = GTX 1050 Ti, sm_86 = A10, sm_89 = L4
GENCODE = -gencode arch=compute_61,code=sm_61 \
          -gencode arch=compute_86,code=sm_86 \
          -gencode arch=compute_89,code=sm_89

# --- Precision mode --------------------------------------------------------
ifeq ($(MCMC_DOUBLE),1)
  CXXFLAGS += -DMCMC_DOUBLE
  export MCMC_DOUBLE=1
endif

.PHONY: all clean purge install

all: libmcmcuda.a list_gpus

libmcmcuda.a: $(OBJECTS)
	$(CXX) -Xcompiler -Wl, -Xcompiler -no-undefined -lib $(LIBS) \
	       $(OBJECTS) -o libmcmcuda.a

mcmcjob.o: mcmcjob.cu mcmcjob.cuh mcmc.h mcmc_types.h
	$(CXX) $(CXXFLAGS) $(GENCODE) -I. -c mcmcjob.cu -o mcmcjob.o

gpu_mcmc.o: gpu_mcmc.cu mcmcjob.cuh mcmc.h mcmc_types.h
	$(CXX) $(CXXFLAGS) $(GENCODE) -I. -I$(MODELDIR) -c gpu_mcmc.cu -o gpu_mcmc.o

scatter_fits: scatter_fits.c
	mkdir -p ~/bin
	$(CC) -o scatter_fits scatter_fits.c -lm -lcfitsio -lfftw3 && \
	cp scatter_fits ~/bin

deblur_uv: deblur_uv.c
	mkdir -p ~/bin
	$(CC) -o deblur_uv deblur_uv.c -lm && \
	cp deblur_uv ~/bin

list_gpus: list_gpus.cu
	mkdir -p ~/bin
	$(CXX) list_gpus.cu -o list_gpus && \
	cp list_gpus ~/bin

install:
	mkdir -p ~/lib64/python
	cp mcmc_interf.*.so ~/lib64/python
	cp imgpu.py ~/lib64/python
	cp obsdatafit.py ~/lib64/python

clean:
	rm -fR lib*.a *.o mcmc_interf.c build/

purge: clean
	rm -fR mcmc_interf.*.so *.pyc scatter_fits deblur_uv list_gpus