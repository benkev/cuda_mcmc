#
# mcmc_interf_float.pyx
#
# Cython interface for cuda_mcmc in FLOAT precision (mcmc_real = float).
# This file is used when MCMC_DOUBLE is NOT set.
#
# See mcmc_interf_double.pyx for the double-precision variant.
#

import numpy as np
cimport numpy as np

# Flag for imgpu.py: this is the float build
MCMC_DOUBLE_FLAG = False

cdef extern from "cuda_runtime_api.h":
    void cudaDeviceReset()

cdef extern from "cuda_runtime_api.h":
    int cudaSetDevice(int devNum)

cdef extern from "mcmc.h":
    void mcmc_cuda(float *coor, float *dat, float *std2r,
            int *icoor, int *idat, float *datm, float *chi2m,
            int *pdescr, int *ivar, int *invar, float *ptotal,
            float *pmint, float *pmaxt,
            float *pout, int *tout, float *chi2,
            float *chi2c, int *tcur, int *flag,
            int *n_acpt, int *n_exch, int *n_cnt, int *n_hist,
            float *beta, float *pstp, float *ptent, int *ptentn,
            float *pcur, unsigned char *rndst,

            unsigned long long seed, int imodel,
            int ncoor, int ndat, int nicoor, int nidat, int ndatm,
            int nptot, int nprm, int nadj, int npass,
            int nbeta, int nseq, int nburn, int niter)


cpdef mcmcuda(
    # Coordinates, Observables and Model Data, Integer Data
    np.ndarray[np.float32_t,ndim=1] coor, #/* [ncoor]: Coordinates: UV etc */
    np.ndarray[np.float32_t,ndim=1] dat,  #/* [ndat]:  Observation data */
    np.ndarray[np.float32_t,ndim=1] std2r,#/* [ndat]:  Data std^2 recipr. */
    np.ndarray[np.int32_t,ndim=1]   icoor,#/* [nicoor]: Any integer coords */
    np.ndarray[np.int32_t,ndim=1]   idat, #/* [nidat]: Any integer data */
    np.ndarray[np.float32_t,ndim=1] datm, #/* [ndatm]: Model data */
    np.ndarray[np.float32_t,ndim=3] chi2m,#/* [nbeta][nseq][ndc2]: chi^2 terms*/
    # Parameters
    np.ndarray[np.int32_t,ndim=1]   pdescr,  #/* [nptot] */
    np.ndarray[np.int32_t,ndim=1]   ivar,    #/* [nprm] */
    np.ndarray[np.int32_t,ndim=1]   invar,   #/* [nptot] */
    np.ndarray[np.float32_t,ndim=1] ptotal,  #/* [nptot] */
    np.ndarray[np.float32_t,ndim=1] pmint,   #/* [nptot] */
    np.ndarray[np.float32_t,ndim=1] pmaxt,   #/* [nptot] */
    # MCMC Algorithm
    np.ndarray[np.float32_t,ndim=4] pout,   #/* [nprm][nbeta][nseq][niter] */
    np.ndarray[np.int32_t,ndim=3]   tout,   #/* [nbeta][nseq][niter] */
    np.ndarray[np.float32_t,ndim=2] chi2,   #/* [nseq][niter] */
    np.ndarray[np.float32_t,ndim=2] chi2c,  #/* [nbeta][nseq] */
    np.ndarray[np.int32_t,ndim=2]   tcur,   #/* [nbeta][nseq] */
    np.ndarray[np.int32_t,ndim=2]   flag,   #/* [nbeta][nseq] */
    np.ndarray[np.int32_t,ndim=3]   n_acpt, #/* [nprm][nbeta][nseq] */
    np.ndarray[np.int32_t,ndim=2]   n_exch, #/* [nbeta][nseq] */
    np.ndarray[np.int32_t,ndim=3]   n_cnt,  #/* [nprm][nbeta][nseq] */
    np.ndarray[np.int32_t,ndim=4]   n_hist, #/* [nadj][nprm][nbeta][nseq] */
    np.ndarray[np.float32_t,ndim=1] beta,   #/* [nbeta] */
    np.ndarray[np.float32_t,ndim=3] pstp,   #/* [nprm][nbeta][nseq] */
    np.ndarray[np.float32_t,ndim=2] ptent,  #/* [nbeta][nseq] */
    np.ndarray[np.int32_t,ndim=2]   ptentn, #/* [nbeta][nseq] */
    np.ndarray[np.float32_t,ndim=3] pcur,   #/* [nbeta][nseq][nptot] */
    # Random Number Generator States
    np.ndarray[np.uint8_t,ndim=3]   rndst,  #/* [nbeta][nseq][48] */

    unsigned long long seed, int imodel,
    int ncoor, int ndat, int nicoor, int nidat, int ndatm,
    int nptot, int nprm, int nadj, int npass,
    int nbeta, int nseq, int nburn, int niter):

    mcmc_cuda(<float*>coor.data,   <float*>dat.data,    <float*>std2r.data,
              <int*>icoor.data,    <int*>idat.data,     <float*>datm.data,
              <float*>chi2m.data,  <int*>pdescr.data,   <int*>ivar.data,
              <int*>invar.data,    <float*>ptotal.data, <float*>pmint.data,
              <float*>pmaxt.data,  <float*>pout.data,   <int*>tout.data,
              <float*>chi2.data,   <float*>chi2c.data,  <int*>tcur.data,
              <int*>flag.data,     <int*>n_acpt.data,   <int*>n_exch.data,
              <int*>n_cnt.data,    <int*>n_hist.data,   <float*>beta.data,
              <float*>pstp.data,   <float*>ptent.data,  <int*>ptentn.data,
              <float*>pcur.data,   <unsigned char*>rndst.data,

              seed, imodel, ncoor, ndat, nicoor, nidat, ndatm, nptot, nprm,
              nadj, npass, nbeta, nseq, nburn, niter)


cdef extern from "mcmc.h":
    void calc_modchi2(float *coor, float *dat, float *std2r,
           int *icoor, int *idat, float *datm, float *chi2m,
           float *pcur, float *chi2c, int *flag, int imodel,
           int ncoor, int ndat, int nicoor, int nidat, int ndatm,
           int nptot, int nbeta, int nseq)


cpdef calcmodchi2(
    np.ndarray[np.float32_t,ndim=1] coor,
    np.ndarray[np.float32_t,ndim=1] dat,
    np.ndarray[np.float32_t,ndim=1] std2r,
    np.ndarray[np.int32_t,ndim=1]   icoor,
    np.ndarray[np.int32_t,ndim=1]   idat,
    np.ndarray[np.float32_t,ndim=1] datm,
    np.ndarray[np.float32_t,ndim=3] chi2m,
    np.ndarray[np.float32_t,ndim=3] pcur,
    np.ndarray[np.float32_t,ndim=2] chi2c,
    np.ndarray[np.int32_t,ndim=2]   flag,
    int imodel, int ncoor, int ndat, int nicoor, int nidat, int ndatm,
    int nptot, int nbeta, int nseq):

    calc_modchi2(<float*>coor.data,   <float*>dat.data,    <float*>std2r.data,
                 <int*>icoor.data,    <int*>idat.data,     <float*>datm.data,
                 <float*>chi2m.data,  <float*>pcur.data,   <float*>chi2c.data,
                 <int*>flag.data,
                 imodel, ncoor, ndat, nicoor, nidat, ndatm, nptot, nbeta, nseq)


cdef extern from "mcmc.h":
    void calc_model(float *coor, int *idat, float *datm, float *pcur,
                    int imodel, int ncoor, int ndat, int nidat,
                    int ndatm, int nptot, int nbeta, int nseq)


cpdef calcmodel(
    np.ndarray[np.float32_t,ndim=1] coor,
    np.ndarray[np.int32_t,ndim=1]   idat,
    np.ndarray[np.float32_t,ndim=1] datm,
    np.ndarray[np.float32_t,ndim=3] pcur,
    int imodel, int ncoor, int ndat, int nidat, int ndatm,
    int nptot, int nbeta, int nseq):

    calc_model(<float*>coor.data, <int*>idat.data, <float*>datm.data,
               <float*>pcur.data,
               imodel, ncoor, ndat, nidat, ndatm, nptot, nbeta, nseq)


cpdef reset_gpu():
    cudaDeviceReset()

cpdef int set_gpu(int devNum):
    return cudaSetDevice(devNum)