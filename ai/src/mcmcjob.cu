//
// mcmcjob.cu
//
// The classes CCalcModel and its inheritor CMcmcFit contain the data
// transferred from the PC memory (host) to the CUDA GPU memory (device).
// The instantiation of the latter and calling the method
// CMcmcFit::do_mcmc_on_gpu() runs the MCMC fitting on a CUDA GPU.
// is made over the interface Python-C/C++ mcmc_interf.pyx written in
// Cython and compiled into mcmc_interf.c.
//
// Types: mcmc_real for critical arrays (dat, datm, chi2m, chi2c, chi2,
// pcur, pout, ptent, pstp, beta, std2r, coor, ptotal, pmint, pmaxt),
// mcmc_int for indices/counters/flags, mcmc_rng for RNG state.
// Switch float/double with -DMCMC_DOUBLE.
//
// (See the long header comment in the previous version for the full
//  description of all parameters and arrays.)
//

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include "mcmcjob.cuh"
#include "mcmc.h"


void cudaAssert(const cudaError err, const char *file, const int line)
{
    if( cudaSuccess != err) {
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",
                file, line, cudaGetErrorString(err) );
        exit(1);
    }
}


/*=========================================================================*/


extern "C"
void calc_modchi2(mcmc_real *coor, mcmc_real *dat, mcmc_real *std2r,
                  mcmc_int *icoor, mcmc_int *idat, mcmc_real *datm,
                  mcmc_real *chi2m,
                  mcmc_real *pcur, mcmc_real *chi2c, mcmc_int *flag,
                  int imodel,
                  int ncoor, int ndat, int nicoor, int nidat, int ndatm,
                  int nptot, int nbeta, int nseq) {

  /*
   * Instantiate the CCalcModel class as modobj object:
   */
  CCalcModel modobj(imodel, ncoor, ndat, nicoor, nidat, ndatm,
                    nptot, nbeta, nseq,
                    /* Coordinates, Observables, Integer Data  */
                    coor,    /* [ncoor] */
                    dat,     /* [ndat] */
                    std2r,   /* [ndat] */
                    icoor,   /* [nicoor] */
                    idat,    /* [nidat] */
                    datm,    /* [ndatm] */
                    chi2m,   /* [nbeta][nseq][ndat] */
                    /* Parameters */
                    pcur,    /* [nbeta][nseq][nptot] */
                    flag,    /* [nbeta][nseq] */
                    chi2c);  /* [nbeta][nseq] */

  modobj.do_model_chi2_gpu();

} /* extern "C" void calc_modchi2() */


extern "C"
void calc_model(mcmc_real *coor, mcmc_int *idat, mcmc_real *datm,
                mcmc_real *pcur,
                int imodel, int ncoor, int ndat, int nidat,
                int ndatm, int nptot, int nbeta, int nseq) {

  /*
   * Arrays below are not used by calc_model() itself, but the CCalcModel
   * constructor requires them (it is shared with calc_modchi2() and
   * mcmc_cuda()). We allocate minimal buffers just to satisfy the
   * constructor; they are never read or written meaningfully.
   */
  mcmc_real *dat = new mcmc_real [ndat];
  mcmc_real *std2r = new mcmc_real [ndat];
  mcmc_real *chi2m = new mcmc_real [nbeta*nseq*ndat];
  mcmc_real *chi2c = new mcmc_real [nbeta*nseq];
  int nicoor = 1;
  mcmc_int *icoor = new mcmc_int [nicoor];
  mcmc_int *flag = new mcmc_int [nbeta*nseq];

  /*
   * Instantiate the CCalcModel class as modobj object:
   */
  CCalcModel modobj(imodel, ncoor, ndat, nicoor, nidat, ndatm,
                    nptot, nbeta, nseq,
                    /* Coordinates, Observables, Integer Data  */
                    coor,    /* [ncoor] */
                    dat,     /* [ndat] */
                    std2r,   /* [ndat] */
                    icoor,   /* [nicoor] */
                    idat,    /* [nidat] */
                    datm,    /* [ndatm] */
                    chi2m,   /* [nbeta][nseq][ndat] */
                    /* Parameters */
                    pcur,    /* [nbeta][nseq][nptot] */
                    flag,    /* [nbeta][nseq] */
                    chi2c);  /* [nbeta][nseq] */

  modobj.do_calc_model_gpu();

  delete[] dat;
  delete[] std2r;
  delete[] chi2m;
  delete[] chi2c;
  delete[] icoor;
  delete[] flag;

} /* extern "C" void calc_model() */


/*
 * Constructors
 */

CCalcModel::CCalcModel(int imodel,
           int ncoor,
           int ndat,
           int nicoor,
           int nidat,
           int ndatm,
           int nptot,
           int nbeta,
           int nseq,
           /* Coordinates, Observables and Model Data, Integer Data */
           mcmc_real *coor,   /* [ncoor]:             Coordinates: UV etc */
           mcmc_real *dat,    /* [ndat]:              Observation data */
           mcmc_real *std2r,  /* [ndat]:              Data std^2 recipr. */
           mcmc_int  *icoor,  /* [nicoor]:            Any integer data */
           mcmc_int  *idat,   /* [nidat]:             Any integer data */
           mcmc_real *datm,   /* [ndatm]: Model data */
           mcmc_real *chi2m,  /* [nbeta][nseq][ndat]: Terms of chi^2 sum */
           /* Parameters */
           mcmc_real *pcur,   /* [nbeta][nseq][nptot] */
           mcmc_int  *flag,   /* [nbeta][nseq] */
           mcmc_real *chi2c)  /* [nbeta][nseq] */  {

  s_coor  = sizeof(mcmc_real)*ncoor;
  s_dat   = sizeof(mcmc_real)*ndat;
  s_icoor = sizeof(mcmc_int)*nicoor;
  s_idat  = sizeof(mcmc_int)*nidat;
  s_datm  = sizeof(mcmc_real)*ndatm;
  s_chi2m = sizeof(mcmc_real)*nbeta*nseq*ndat;
  s_pcur  = sizeof(mcmc_real)*nbeta*nseq*nptot;
  s_bsint = sizeof(mcmc_int)*nbeta*nseq;
  s_bsflt = sizeof(mcmc_real)*nbeta*nseq;

  this->imodel = imodel;
  this->ncoor = ncoor;
  this->ndat = ndat;
  this->nicoor = nicoor;
  this->nidat = nidat;
  this->ndatm = ndatm;
  this->nptot = nptot;
  this->nbeta = nbeta;
  this->nseq = nseq;
  /* Coordinates, Observables and Model Data, Integer Data */
  this->coor_h = coor;       /* [ncoor] */
  this->dat_h = dat;         /* [ndat] */
  this->std2r_h = std2r;
  this->icoor_h = icoor;
  this->idat_h = idat;
  this->datm_h = datm;
  this->chi2m_h = chi2m;
  /* Parameters */
  this->pcur_h = pcur;       /* [nbeta][nseq][nptot] */
  this->flag_h = flag;       /* [nbeta][nseq] */
  this->chi2c_h = chi2c;     /* [nbeta][nseq] */

  this->nchi2m = nbeta*nseq*ndat; /* Threads in model data calculation */


  /*
   * Allocate the arrays on the GPU
   */
  CUDA_CALL(cudaMalloc((void **) &this->coor,  s_coor));
  CUDA_CALL(cudaMalloc((void **) &this->dat,   s_dat));
  CUDA_CALL(cudaMalloc((void **) &this->std2r, s_dat));
  CUDA_CALL(cudaMalloc((void **) &this->icoor, s_icoor));
  CUDA_CALL(cudaMalloc((void **) &this->idat,  s_idat));
  CUDA_CALL(cudaMalloc((void **) &this->datm,  s_datm));
  CUDA_CALL(cudaMalloc((void **) &this->chi2m, s_chi2m));
  CUDA_CALL(cudaMalloc((void **) &this->pcur,  s_pcur));
  CUDA_CALL(cudaMalloc((void **) &this->flag,  s_bsint));
  CUDA_CALL(cudaMalloc((void **) &this->chi2c, s_bsflt));

  /*
   * Copy the host arrays to the GPU
   */
  CUDA_CALL(cudaMemcpy(this->coor,  this->coor_h,  s_coor,  HtoD));
  CUDA_CALL(cudaMemcpy(this->dat,   this->dat_h,   s_dat,   HtoD));
  CUDA_CALL(cudaMemcpy(this->std2r, this->std2r_h, s_dat,   HtoD));
  CUDA_CALL(cudaMemcpy(this->icoor, this->icoor_h, s_icoor, HtoD));
  CUDA_CALL(cudaMemcpy(this->idat,  this->idat_h,  s_idat,  HtoD));
  CUDA_CALL(cudaMemcpy(this->datm,  this->datm_h,  s_datm,  HtoD));
  CUDA_CALL(cudaMemcpy(this->chi2m, this->chi2m_h, s_chi2m, HtoD));
  CUDA_CALL(cudaMemcpy(this->pcur,  this->pcur_h,  s_pcur,  HtoD));
  CUDA_CALL(cudaMemcpy(this->flag,  this->flag_h,  s_bsint, HtoD));
  CUDA_CALL(cudaMemcpy(this->chi2c, this->chi2c_h, s_bsflt, HtoD));

  printf("CONSTRUCTOR of CCalcModel: exit.\n");
}


/*
 * Destructor
 */

CCalcModel::~CCalcModel() {

  CUDA_CALL(cudaMemcpy(this->coor_h,  this->coor,  s_coor,  DtoH));
  CUDA_CALL(cudaMemcpy(this->dat_h,   this->dat,   s_dat,   DtoH));
  CUDA_CALL(cudaMemcpy(this->std2r_h, this->std2r, s_dat,   DtoH));
  CUDA_CALL(cudaMemcpy(this->icoor_h, this->icoor, s_icoor, DtoH));
  CUDA_CALL(cudaMemcpy(this->idat_h,  this->idat,  s_idat,  DtoH));
  CUDA_CALL(cudaMemcpy(this->datm_h,  this->datm,  s_datm,  DtoH));
  CUDA_CALL(cudaMemcpy(this->chi2m_h, this->chi2m, s_chi2m, DtoH));
  CUDA_CALL(cudaMemcpy(this->pcur_h,  this->pcur,  s_pcur,  DtoH));
  CUDA_CALL(cudaMemcpy(this->flag_h,  this->flag,  s_bsint, DtoH));
  CUDA_CALL(cudaMemcpy(this->chi2c_h, this->chi2c, s_bsflt, DtoH));

  CUDA_CALL(cudaFree(this->coor));
  CUDA_CALL(cudaFree(this->dat));
  CUDA_CALL(cudaFree(this->std2r));
  CUDA_CALL(cudaFree(this->icoor));
  CUDA_CALL(cudaFree(this->idat));
  CUDA_CALL(cudaFree(this->datm));
  CUDA_CALL(cudaFree(this->chi2m));
  CUDA_CALL(cudaFree(this->pcur));
  CUDA_CALL(cudaFree(this->flag));
  CUDA_CALL(cudaFree(this->chi2c));

}



void CCalcModel::do_model_chi2_gpu() {
  CCalcModel *dev_mod;

  CUDA_CALL(cudaMalloc((void **) &dev_mod, sizeof(CCalcModel)));
  CUDA_CALL(cudaMemcpy(dev_mod, this, sizeof(CCalcModel), HtoD));

  run_calcmodchi2(this, dev_mod);

  CUDA_CALL(cudaMemcpy(this, dev_mod, sizeof(CCalcModel), DtoH));
  CUDA_CALL(cudaFree(dev_mod));

}



void CCalcModel::do_calc_model_gpu() {
  CCalcModel *dev_mod;

  CUDA_CALL(cudaMalloc((void **) &dev_mod, sizeof(CCalcModel)));
  CUDA_CALL(cudaMemcpy(dev_mod, this, sizeof(CCalcModel), HtoD));

  run_calcmodel(this, dev_mod);

  CUDA_CALL(cudaMemcpy(this, dev_mod, sizeof(CCalcModel), DtoH));
  CUDA_CALL(cudaFree(dev_mod));

}


/*=========================================================================*/


extern "C"
void mcmc_cuda(mcmc_real *coor, mcmc_real *dat, mcmc_real *std2r,
               mcmc_int *icoor, mcmc_int *idat, mcmc_real *datm,
               mcmc_real *chi2m,
               mcmc_int *pdescr, mcmc_int *ivar, mcmc_int *invar,
               mcmc_real *ptotal, mcmc_real *pmint, mcmc_real *pmaxt,
               mcmc_real *pout, mcmc_int *tout, mcmc_real *chi2,
               mcmc_real *chi2c, mcmc_int *tcur, mcmc_int *flag,
               mcmc_int *n_acpt, mcmc_int *n_exch, mcmc_int *n_cnt,
               mcmc_int *n_hist,
               mcmc_real *beta, mcmc_real *pstp, mcmc_real *ptent,
               mcmc_int *ptentn,
               mcmc_real *pcur, mcmc_rng *rndst,

               unsigned long long seed, int imodel,
               int ncoor, int ndat, int nicoor, int nidat, int ndatm,
               int nptot, int nprm, int nadj, int npass,
               int nbeta, int nseq, int nburn, int niter) {

  /*
   * Instantiate the CMcmcFit class as mcmcobj object:
   */
  CMcmcFit mcmcobj(imodel,
                   ncoor,
                   ndat,
                   nicoor,
                   nidat,
                   ndatm,
                   nptot,
                   nprm,
                   nadj,
                   npass,
                   nbeta,
                   nseq,
                   nburn,
                   niter,
                   /* Random Number Generators */
                   seed,
                   rndst,   /* [nbeta][nseq][48] */
                   /* Coordinates, Observables, Integer Data  */
                   coor,    /* [ncoor] */
                   dat,     /* [ndat] */
                   std2r,   /* [ndat] */
                   icoor,   /* [nicoor] */
                   idat,    /* [nidat] */
                   datm,    /* [ndatm] */
                   chi2m,   /* [nbeta][nseq][ndat] */
                   /* Parameters */
                   pdescr,  /* [nptot] */
                   ivar,    /* [nprm] */
                   invar,   /* [nptot] */
                   ptotal,  /* [nptot] */
                   pmint,   /* [nptot] */
                   pmaxt,   /* [nptot] */
                   /* MCMC Algorithm */
                   pout,    /* [nprm][nbeta][nseq][niter] */
                   tout,    /* [nbeta][nseq][niter] */
                   chi2,    /* [nseq][niter] */
                   chi2c,   /* [nbeta][nseq] */
                   tcur,    /* [nbeta][nseq] */
                   flag,    /* [nbeta][nseq] */
                   n_acpt,  /* [nprm][nbeta][nseq] */
                   n_exch,  /* [nbeta][nseq] */
                   n_cnt,   /* [nprm][nbeta][nseq] */
                   n_hist,  /* [nadj][nprm][nbeta][nseq] */
                   beta,    /* [nbeta] */
                   pstp,    /* [nprm][nbeta][nseq] */
                   ptent,   /* [nbeta][nseq] */
                   ptentn,  /* [nbeta][nseq] */
                   pcur);   /* [nbeta][nseq][nptot] */

  mcmcobj.do_mcmc_on_gpu();

}  /* extern "C" void mcmc_cuda() */


/*
 * Constructor of CMcmcFit
 */

CMcmcFit::CMcmcFit(int imodel,
                   int ncoor,
                   int ndat,
                   int nicoor,
                   int nidat,
                   int ndatm,
                   int nptot,
                   int nprm,
                   int nadj,
                   int npass,
                   int nbeta,
                   int nseq,
                   int nburn,
                   int niter,
                   /* Random Number Generators */
                   unsigned long long seed,
                   mcmc_rng *rndst,
                   /* Coordinates, Observables and Model Data, Integer Data */
                   mcmc_real *coor,   /* [ncoor]:             Coordinates: UV etc */
                   mcmc_real *dat,    /* [ndat]:              Observation data */
                   mcmc_real *std2r,  /* [ndat]:              Data std^2 recipr. */
                   mcmc_int  *icoor,  /* [nicoor]:            Any integer data */
                   mcmc_int  *idat,   /* [nidat]:             Any integer data */
                   mcmc_real *datm,   /* [ndatm]: Model data */
                   mcmc_real *chi2m,  /* [nbeta][nseq][ndat]: Terms of chi^2 sum */
                   /* Parameters */
                   mcmc_int  *pdescr, /* [nptot] */
                   mcmc_int  *ivar,   /* nprm] */
                   mcmc_int  *invar,  /* [nptot] */
                   mcmc_real *ptotal, /* [nptot] */
                   mcmc_real *pmint,  /* [nptot] */
                   mcmc_real *pmaxt,  /* [nptot] */
                   /* MCMC Algorithm */
                   mcmc_real *pout,   /* [nprm][nbeta][nseq][niter] */
                   mcmc_int  *tout,   /* [nbeta][nseq][niter] */
                   mcmc_real *chi2,   /* [nseq][niter] */
                   mcmc_real *chi2c,  /* [nbeta][nseq] */
                   mcmc_int  *tcur,   /* [nbeta][nseq] */
                   mcmc_int  *flag,   /* [nbeta][nseq] */
                   mcmc_int  *n_acpt, /* [nprm][nbeta][nseq] */
                   mcmc_int  *n_exch, /* [nbeta][nseq] */
                   mcmc_int  *n_cnt,  /* [nprm][nbeta][nseq] */
                   mcmc_int  *n_hist, /* [nadj][nprm][nbeta][nseq] */
                   mcmc_real *beta,   /* [nbeta] */
                   mcmc_real *pstp,   /* [nprm][nbeta][nseq] */
                   mcmc_real *ptent,  /* [nbeta][nseq] */
                   mcmc_int  *ptentn, /* [nbeta][nseq] */
                   mcmc_real *pcur)   /* [nbeta][nseq][nptot] */
{

  ntitr = nburn + niter;

  s_coor   = sizeof(mcmc_real)*ncoor;
  s_dat    = sizeof(mcmc_real)*ndat;
  s_datm   = sizeof(mcmc_real)*ndatm;
  s_icoor  = sizeof(mcmc_int)*nicoor;
  s_idat   = sizeof(mcmc_int)*nidat;
  s_rndst  = sizeof(curandState)*nbeta*nseq;
  s_ptot   = sizeof(mcmc_real)*nptot;
  s_prm    = sizeof(mcmc_int)*nprm;
  s_pout   = sizeof(mcmc_real)*nprm*nbeta*nseq*niter;
  s_tout   = sizeof(mcmc_int)*nbeta*nseq*niter;
  s_chi2   = sizeof(mcmc_real)*nseq*niter;
  s_beta   = sizeof(mcmc_real)*nbeta;
  s_acpt   = sizeof(mcmc_int)*nprm*nbeta*nseq;
  s_hist   = sizeof(mcmc_int)*nadj*nprm*nbeta*nseq;
  s_bsflt  = sizeof(mcmc_real)*nbeta*nseq;
  s_bsint  = sizeof(mcmc_int)*nbeta*nseq;
  s_chi2m  = sizeof(mcmc_real)*nbeta*nseq*ndat;
  s_pcur   = sizeof(mcmc_real)*nbeta*nseq*nptot;
  s_bsptot = sizeof(mcmc_real)*nbeta*nseq*nptot;
  s_4bstf  = sizeof(mcmc_real)*nbeta*nseq*4*ntitr;
  s_bstf   = sizeof(mcmc_real)*nbeta*nseq*ntitr;
  s_bsti   = sizeof(mcmc_int)*nbeta*nseq*ntitr;
  s_burn   = sizeof(mcmc_real)*nprm*nbeta*nseq*nburn;

  this->imodel = imodel;
  this->ncoor = ncoor;
  this->ndat = ndat;
  this->nicoor = nicoor;
  this->nidat = nidat;
  this->ndatm = ndatm;
  this->nptot = nptot;
  this->nprm = nprm;
  this->nadj = nadj;
  this->npass = npass;
  this->nbeta = nbeta;
  this->nseq = nseq;
  this->nburn = nburn;
  this->niter = niter;
  /* Random Number Generators */
  this->seed = seed;
  this->rndst_h = (curandState *) rndst;
  /* Coordinates, Observables and Model Data, Integer Data */
  this->coor_h = coor;       /* [ncoor] */
  this->dat_h = dat;         /* [ndat] */
  this->std2r_h = std2r;
  this->icoor_h = icoor;
  this->idat_h = idat;
  this->datm_h = datm;
  this->chi2m_h = chi2m;
  /* Parameters */
  this->pdescr_h = pdescr;
  this->ivar_h = ivar;
  this->invar_h = invar;
  this->ptotal_h = ptotal;
  this->pmint_h = pmint;
  this->pmaxt_h = pmaxt;
  /* MCMC Algorithm */
  this->pout_h = pout;
  this->tout_h = tout;
  this->chi2_h = chi2;       /* [nseq][niter] */
  this->chi2c_h = chi2c;     /* [nbeta][nseq] */
  this->tcur_h = tcur;       /* [nbeta][nseq] */
  this->flag_h = flag;       /* [nbeta][nseq] */
  this->n_acpt_h = n_acpt;   /* [nprm][nbeta][nseq] */
  this->n_exch_h = n_exch;   /* [nbeta][nseq] */
  this->n_cnt_h = n_cnt;     /* [nprm][nbeta][nseq] */
  this->n_hist_h = n_hist;   /* [nadj][nprm][nbeta][nseq] */
  this->beta_h = beta;       /* [nbeta] */
  this->pstp_h = pstp;       /* [nprm][nbeta][nseq] */
  this->ptent_h = ptent;     /* [nbeta][nseq] */
  this->ptentn_h = ptentn;   /* [nbeta][nseq] */
  this->pcur_h = pcur;       /* [nbeta][nseq][nptot] */

  this->ipm = 0; /* Loop parameter: current model parameter */
  this->itr = 0; /* Loop parameter: current iteration */
  this->nrndst = nbeta*nseq;
  this->nchi2m = nbeta*nseq*ndat; /* Threads in model data calculation */


  /*
   * Allocate the arrays on the GPU
   */

  printf("s_rndst = %d\n", s_rndst);

  CUDA_CALL(cudaMalloc((void **) &this->rndst, s_rndst));

  CUDA_CALL(cudaMalloc((void **) &this->coor, s_coor));
  CUDA_CALL(cudaMalloc((void **) &this->dat, s_dat));
  CUDA_CALL(cudaMalloc((void **) &this->std2r, s_dat));
  CUDA_CALL(cudaMalloc((void **) &this->icoor, s_icoor));
  CUDA_CALL(cudaMalloc((void **) &this->idat, s_idat));
  CUDA_CALL(cudaMalloc((void **) &this->datm, s_datm));
  CUDA_CALL(cudaMalloc((void **) &this->chi2m, s_chi2m));
  CUDA_CALL(cudaMalloc((void **) &this->pdescr, s_ptot));
  CUDA_CALL(cudaMalloc((void **) &this->ivar, s_prm));
  CUDA_CALL(cudaMalloc((void **) &this->invar, s_ptot));
  CUDA_CALL(cudaMalloc((void **) &this->ptotal, s_ptot));
  CUDA_CALL(cudaMalloc((void **) &this->pmint, s_ptot));
  CUDA_CALL(cudaMalloc((void **) &this->pmaxt, s_ptot));
  CUDA_CALL(cudaMalloc((void **) &this->pout, s_pout));
  CUDA_CALL(cudaMalloc((void **) &this->tout, s_tout));
  CUDA_CALL(cudaMalloc((void **) &this->chi2, s_chi2));
  CUDA_CALL(cudaMalloc((void **) &this->chi2c, s_bsflt));
  CUDA_CALL(cudaMalloc((void **) &this->tcur, s_bsint));
  CUDA_CALL(cudaMalloc((void **) &this->flag, s_bsint));
  CUDA_CALL(cudaMalloc((void **) &this->n_acpt, s_acpt));
  CUDA_CALL(cudaMalloc((void **) &this->n_exch, s_bsint));
  CUDA_CALL(cudaMalloc((void **) &this->n_cnt, s_acpt));
  CUDA_CALL(cudaMalloc((void **) &this->n_hist, s_hist));
  CUDA_CALL(cudaMalloc((void **) &this->beta, s_beta));
  CUDA_CALL(cudaMalloc((void **) &this->pstp, s_acpt));
  CUDA_CALL(cudaMalloc((void **) &this->ptent, s_bsflt));
  CUDA_CALL(cudaMalloc((void **) &this->ptentn, s_bsint));
  CUDA_CALL(cudaMalloc((void **) &this->pcur, s_bsptot));

  /*
   * Copy the host arrays to the GPU
   */

  CUDA_CALL(cudaMemcpy(this->rndst, this->rndst_h, s_rndst, HtoD));

  CUDA_CALL(cudaMemcpy(this->coor, this->coor_h, s_coor, HtoD));
  CUDA_CALL(cudaMemcpy(this->dat, this->dat_h, s_dat, HtoD));
  CUDA_CALL(cudaMemcpy(this->std2r, this->std2r_h, s_dat, HtoD));
  CUDA_CALL(cudaMemcpy(this->icoor, this->icoor_h, s_icoor, HtoD));
  CUDA_CALL(cudaMemcpy(this->idat, this->idat_h, s_idat, HtoD));
  CUDA_CALL(cudaMemcpy(this->datm, this->datm_h, s_datm, HtoD));
  CUDA_CALL(cudaMemcpy(this->chi2m, this->chi2m_h, s_chi2m, HtoD));
  CUDA_CALL(cudaMemcpy(this->pdescr, this->pdescr_h, s_ptot, HtoD));
  CUDA_CALL(cudaMemcpy(this->ivar, this->ivar_h, s_prm, HtoD));
  CUDA_CALL(cudaMemcpy(this->invar, this->invar_h, s_ptot, HtoD));
  CUDA_CALL(cudaMemcpy(this->ptotal, this->ptotal_h, s_ptot, HtoD));
  CUDA_CALL(cudaMemcpy(this->pmint, this->pmint_h, s_ptot, HtoD));
  CUDA_CALL(cudaMemcpy(this->pmaxt, this->pmaxt_h, s_ptot, HtoD));
  CUDA_CALL(cudaMemcpy(this->tout, this->tout_h, s_tout, HtoD));
  CUDA_CALL(cudaMemcpy(this->chi2, this->chi2_h, s_chi2, HtoD));
  CUDA_CALL(cudaMemcpy(this->chi2c, this->chi2c_h, s_bsflt, HtoD));
  CUDA_CALL(cudaMemcpy(this->pout, this->pout_h, s_pout, HtoD));
  CUDA_CALL(cudaMemcpy(this->tcur, this->tcur_h, s_bsint, HtoD));
  CUDA_CALL(cudaMemcpy(this->flag, this->flag_h, s_bsint, HtoD));
  CUDA_CALL(cudaMemcpy(this->n_acpt, this->n_acpt_h, s_acpt, HtoD));
  CUDA_CALL(cudaMemcpy(this->n_cnt, this->n_cnt_h, s_acpt, HtoD));
  CUDA_CALL(cudaMemcpy(this->n_exch, this->n_exch_h, s_bsint, HtoD));
  CUDA_CALL(cudaMemcpy(this->n_hist, this->n_hist_h, s_hist, HtoD));
  CUDA_CALL(cudaMemcpy(this->beta, this->beta_h, s_beta, HtoD));
  CUDA_CALL(cudaMemcpy(this->pstp, this->pstp_h, s_acpt, HtoD));
  CUDA_CALL(cudaMemcpy(this->ptent, this->ptent_h, s_bsflt, HtoD));
  CUDA_CALL(cudaMemcpy(this->ptentn, this->ptentn_h, s_bsint, HtoD));
  CUDA_CALL(cudaMemcpy(this->pcur, this->pcur_h, s_bsptot, HtoD));

}


/*
 * Destructor
 */


CMcmcFit::~CMcmcFit() {

  CUDA_CALL(cudaMemcpy(this->rndst_h, this->rndst, s_rndst, DtoH));

  CUDA_CALL(cudaMemcpy(this->pdescr_h, this->pdescr, s_ptot, DtoH));
  CUDA_CALL(cudaMemcpy(this->ivar_h, this->ivar, s_prm, DtoH));
  CUDA_CALL(cudaMemcpy(this->invar_h, this->invar, s_ptot, DtoH));
  CUDA_CALL(cudaMemcpy(this->ptotal_h, this->ptotal, s_ptot, DtoH));
  CUDA_CALL(cudaMemcpy(this->pmint_h, this->pmint, s_ptot, DtoH));
  CUDA_CALL(cudaMemcpy(this->pmaxt_h, this->pmaxt, s_ptot, DtoH));

  CUDA_CALL(cudaMemcpy(this->pout_h, this->pout, s_pout, DtoH));
  CUDA_CALL(cudaMemcpy(this->tout_h, this->tout, s_tout, DtoH));
  CUDA_CALL(cudaMemcpy(this->chi2_h, this->chi2, s_chi2, DtoH));
  CUDA_CALL(cudaMemcpy(this->tcur_h, this->tcur, s_bsint, DtoH));
  CUDA_CALL(cudaMemcpy(this->n_acpt_h, this->n_acpt, s_acpt, DtoH));
  CUDA_CALL(cudaMemcpy(this->n_cnt_h, this->n_cnt, s_acpt, DtoH));
  CUDA_CALL(cudaMemcpy(this->n_exch_h, this->n_exch, s_bsint, DtoH));
  CUDA_CALL(cudaMemcpy(this->n_hist_h, this->n_hist, s_hist, DtoH));
  CUDA_CALL(cudaMemcpy(this->beta_h, this->beta, s_beta, DtoH));
  CUDA_CALL(cudaMemcpy(this->pstp_h, this->pstp, s_acpt, DtoH));
  CUDA_CALL(cudaMemcpy(this->ptent_h, this->ptent, s_bsflt, DtoH));
  CUDA_CALL(cudaMemcpy(this->ptentn_h, this->ptentn, s_bsint, DtoH));

  CUDA_CALL(cudaFree(this->rndst));

  CUDA_CALL(cudaFree(this->pdescr));
  CUDA_CALL(cudaFree(this->ivar));
  CUDA_CALL(cudaFree(this->invar));
  CUDA_CALL(cudaFree(this->ptotal));
  CUDA_CALL(cudaFree(this->pmint));
  CUDA_CALL(cudaFree(this->pmaxt));

  CUDA_CALL(cudaFree(this->pout));
  CUDA_CALL(cudaFree(this->tout));
  CUDA_CALL(cudaFree(this->chi2));

  CUDA_CALL(cudaFree(this->tcur));
  CUDA_CALL(cudaFree(this->n_acpt));
  CUDA_CALL(cudaFree(this->n_cnt));
  CUDA_CALL(cudaFree(this->n_exch));
  CUDA_CALL(cudaFree(this->n_hist));
  CUDA_CALL(cudaFree(this->beta));
  CUDA_CALL(cudaFree(this->pstp));
  CUDA_CALL(cudaFree(this->ptent));
  CUDA_CALL(cudaFree(this->ptentn));

}


void CMcmcFit::do_mcmc_on_gpu() {
  CMcmcFit *dev_mcmc;

  CUDA_CALL(cudaMalloc((void **) &dev_mcmc, sizeof(CMcmcFit)));
  CUDA_CALL(cudaMemcpy(dev_mcmc, this, sizeof(CMcmcFit), HtoD));

  run_mcmc(this, dev_mcmc);

  printf("sizeof(CMcmcFit) = %lu B\n", (unsigned long)sizeof(CMcmcFit));

  CUDA_CALL(cudaMemcpy(this, dev_mcmc, sizeof(CMcmcFit), DtoH));
  CUDA_CALL(cudaFree(dev_mcmc));

}
