#
# mcmc_interf_double.pyx
#
# Cython interface for cuda_mcmc in DOUBLE precision (mcmc_real = double).
# This file is used when MCMC_DOUBLE is set.
#
# See mcmc_interf_float.pyx for the float-precision variant.
#

import numpy as np
cimport numpy as np

# Flag for imgpu.py: this is the double build
MCMC_DOUBLE_FLAG = True

cdef extern from "cuda_runtime_api.h":
    void cudaDeviceReset()

cdef extern from "cuda_runtime_api.h":
    int cudaSetDevice(int devNum)

cdef extern from "mcmc.h":
    void mcmc_cuda(double *coor, double *dat, double *std2r,
            int *icoor, int *idat, double *datm, double *chi2m,
            int *pdescr, int *ivar, int *invar, double *ptotal,
            double *pmint, double *pmaxt,
            double *pout, int *tout, double *chi2,
            double *chi2c, int *tcur, int *flag,
            int *n_acpt, int *n_exch, int *n_cnt, int *n_hist,
            double *beta, double *pstp, double *ptent, int *ptentn,
            double *pcur, unsigned char *rndst,

            unsigned long long seed, int imodel,
            int ncoor, int ndat, int nicoor, int nidat, int ndatm,
            int nptot, int nprm, int nadj, int npass,
            int nbeta, int nseq, int nburn, int niter)


cpdef mcmcuda(
    np.ndarray[np.float64_t,ndim=1] coor,
    np.ndarray[np.float64_t,ndim=1] dat,
    np.ndarray[np.float64_t,ndim=1] std2r,
    np.ndarray[np.int32_t,ndim=1]   icoor,
    np.ndarray[np.int32_t,ndim=1]   idat,
    np.ndarray[np.float64_t,ndim=1] datm,
    np.ndarray[np.float64_t,ndim=3] chi2m,
    np.ndarray[np.int32_t,ndim=1]   pdescr,
    np.ndarray[np.int32_t,ndim=1]   ivar,
    np.ndarray[np.int32_t,ndim=1]   invar,
    np.ndarray[np.float64_t,ndim=1] ptotal,
    np.ndarray[np.float64_t,ndim=1] pmint,
    np.ndarray[np.float64_t,ndim=1] pmaxt,
    np.ndarray[np.float64_t,ndim=4] pout,
    np.ndarray[np.int32_t,ndim=3]   tout,
    np.ndarray[np.float64_t,ndim=2] chi2,
    np.ndarray[np.float64_t,ndim=2] chi2c,
    np.ndarray[np.int32_t,ndim=2]   tcur,
    np.ndarray[np.int32_t,ndim=2]   flag,
    np.ndarray[np.int32_t,ndim=3]   n_acpt,
    np.ndarray[np.int32_t,ndim=2]   n_exch,
    np.ndarray[np.int32_t,ndim=3]   n_cnt,
    np.ndarray[np.int32_t,ndim=4]   n_hist,
    np.ndarray[np.float64_t,ndim=1] beta,
    np.ndarray[np.float64_t,ndim=3] pstp,
    np.ndarray[np.float64_t,ndim=2] ptent,
    np.ndarray[np.int32_t,ndim=2]   ptentn,
    np.ndarray[np.float64_t,ndim=3] pcur,
    np.ndarray[np.uint8_t,ndim=3]   rndst,

    unsigned long long seed, int imodel,
    int ncoor, int ndat, int nicoor, int nidat, int ndatm,
    int nptot, int nprm, int nadj, int npass,
    int nbeta, int nseq, int nburn, int niter):

    mcmc_cuda(<double*>coor.data,   <double*>dat.data,    <double*>std2r.data,
              <int*>icoor.data,     <int*>idat.data,      <double*>datm.data,
              <double*>chi2m.data,  <int*>pdescr.data,    <int*>ivar.data,
              <int*>invar.data,     <double*>ptotal.data, <double*>pmint.data,
              <double*>pmaxt.data,  <double*>pout.data,   <int*>tout.data,
              <double*>chi2.data,   <double*>chi2c.data,  <int*>tcur.data,
              <int*>flag.data,      <int*>n_acpt.data,    <int*>n_exch.data,
              <int*>n_cnt.data,     <int*>n_hist.data,    <double*>beta.data,
              <double*>pstp.data,   <double*>ptent.data,  <int*>ptentn.data,
              <double*>pcur.data,   <unsigned char*>rndst.data,

              seed, imodel, ncoor, ndat, nicoor, nidat, ndatm, nptot, nprm,
              nadj, npass, nbeta, nseq, nburn, niter)


cdef extern from "mcmc.h":
    void calc_modchi2(double *coor, double *dat, double *std2r,
           int *icoor, int *idat, double *datm, double *chi2m,
           double *pcur, double *chi2c, int *flag, int imodel,
           int ncoor, int ndat, int nicoor, int nidat, int ndatm,
           int nptot, int nbeta, int nseq)


cpdef calcmodchi2(
    np.ndarray[np.float64_t,ndim=1] coor,
    np.ndarray[np.float64_t,ndim=1] dat,
    np.ndarray[np.float64_t,ndim=1] std2r,
    np.ndarray[np.int32_t,ndim=1]   icoor,
    np.ndarray[np.int32_t,ndim=1]   idat,
    np.ndarray[np.float64_t,ndim=1] datm,
    np.ndarray[np.float64_t,ndim=3] chi2m,
    np.ndarray[np.float64_t,ndim=3] pcur,
    np.ndarray[np.float64_t,ndim=2] chi2c,
    np.ndarray[np.int32_t,ndim=2]   flag,
    int imodel, int ncoor, int ndat, int nicoor, int nidat, int ndatm,
    int nptot, int nbeta, int nseq):

    calc_modchi2(<double*>coor.data,   <double*>dat.data,    <double*>std2r.data,
                 <int*>icoor.data,     <int*>idat.data,      <double*>datm.data,
                 <double*>chi2m.data,  <double*>pcur.data,   <double*>chi2c.data,
                 <int*>flag.data,
                 imodel, ncoor, ndat, nicoor, nidat, ndatm, nptot, nbeta, nseq)


cdef extern from "mcmc.h":
    void calc_model(double *coor, int *idat, double *datm, double *pcur,
                    int imodel, int ncoor, int ndat, int nidat,
                    int ndatm, int nptot, int nbeta, int nseq)


cpdef calcmodel(
    np.ndarray[np.float64_t,ndim=1] coor,
    np.ndarray[np.int32_t,ndim=1]   idat,
    np.ndarray[np.float64_t,ndim=1] datm,
    np.ndarray[np.float64_t,ndim=3] pcur,
    int imodel, int ncoor, int ndat, int nidat, int ndatm,
    int nptot, int nbeta, int nseq):

    calc_model(<double*>coor.data, <int*>idat.data, <double*>datm.data,
               <double*>pcur.data,
               imodel, ncoor, ndat, nidat, ndatm, nptot, nbeta, nseq)


cpdef reset_gpu():
    cudaDeviceReset()

cpdef int set_gpu(int devNum):
    return cudaSetDevice(devNum)