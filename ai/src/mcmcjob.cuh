/* mcmcjob.cuh */

#ifndef MCMCJOB_CUH
#define MCMCJOB_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "mcmc_types.h"

/* #define CUDA_CALL(x) do { if((x) != cudaSuccess) {	\ */
/*       printf("Error at %s:%d\n",__FILE__,__LINE__);	\ */
/*       exit(EXIT_FAILURE);}} while(0) */
#define CUDA_CALL(call) { cudaAssert(call,__FILE__,__LINE__); }


#define min(a, b) ((a) < (b) ? (a) : (b))
#define max(a, b) ((a) > (b) ? (a) : (b))

#define HtoD cudaMemcpyHostToDevice
#define DtoH cudaMemcpyDeviceToHost

/* pi/pi4 теперь mcmc_real, независимо от MCMC_DOUBLE */
#define pi  ((mcmc_real)3.1415926535897931)
#define pi4 ((mcmc_real)12.566370614359172) /* 4*pi */

#ifdef __cplusplus


/*====================================================================*/
/*                BASE CLASS CCalcModel                               */


class CCalcModel {
 public:

  int imodel;
  int ncoor;
  int ndat;
  int nicoor;
  int nidat;
  int ndatm;
  int nptot; /* Keep 'nptot' because run_calcmodchi2() uses it */
  int nbeta;
  int nseq;
  int nchi2m; /* Number of threads in model calculation */

  /*
   * Array sizes (in bytes). For very large arrays these should be
   * size_t, but for the current problem sizes int is sufficient.
   */
  int s_coor;   /* = sizeof(mcmc_real)*ncoor; */
  int s_dat;    /* = sizeof(mcmc_real)*ndat; */
  int s_icoor;  /* = sizeof(mcmc_int)*nicoor; */
  int s_idat;   /* = sizeof(mcmc_int)*nidat; */
  int s_datm;   /* = sizeof(mcmc_real)*ndatm; */
  int s_chi2m;  /* = sizeof(mcmc_real)*ndatm; */
  int s_pcur;   /* = sizeof(mcmc_real)*nbeta*nseq*nptot; */
  int s_bsint;  /* = sizeof(mcmc_int)*nbeta*nseq; */
  int s_bsflt;  /* = sizeof(mcmc_real)*nbeta*nseq; */

  /*
   * DEVICE pointers to data.
   * nseq is the number of streaming multiprocessors.
   */
  /* Coordinates, Observables and Model Data, Integer Data */
  mcmc_real *coor;   /* [ncoor]:             Coordinates: UV etc */
  mcmc_real *dat;    /* [ndat]:              Observation data */
  mcmc_real *std2r;  /* [ndat]:              Data std^2 recipr. */
  mcmc_int  *icoor;  /* [nicoor]:            Any integer coordinates */
  mcmc_int  *idat;   /* [nidat]:             Any integer data */
  mcmc_real *datm;   /* [ndatm]:             Model data */
  mcmc_real *chi2m;  /* [nbeta][nseq][ndat]: Terms of chi^2 sum in datm*/
  /* Parameters */
  mcmc_real *pcur;   /* [nbeta][nseq][nptot] */
  mcmc_int  *flag;   /* [nbeta][nseq] */
  mcmc_real *chi2c;  /* [nbeta][nseq] */

  /*
   * HOST Pointers to data
   */
  mcmc_real *coor_h;   /* [ncoor]:             Coordinates: UV etc */
  mcmc_real *dat_h;    /* [ndat]:              Observation data */
  mcmc_real *std2r_h;  /* [ndat]:              Data std^2 recipr. */
  mcmc_int  *icoor_h;  /* [nicoor]:            Any integer data */
  mcmc_int  *idat_h;   /* [nidat]:             Any integer data */
  mcmc_real *datm_h;   /* [ndatm]: Model data */
  mcmc_real *chi2m_h;  /* [nbeta][nseq][ndat]: Terms of chi^2 sum in datm_h*/
  /* Parameters */
  mcmc_real *pcur_h;   /* [nbeta][nseq][nptot] */
  mcmc_int  *flag_h;   /* [nbeta][nseq] */
  mcmc_real *chi2c_h;  /* [nbeta][nseq] */


  CCalcModel() {}      /* Default constructor */

  CCalcModel(int imodel,
             int ncoor, int ndat, int nicoor, int nidat, int ndatm,
             int nptot, int nbeta, int nseq,
             /* Coordinates, Observables and Model Data, Integer Data */
             mcmc_real *coor,   /* [ncoor] */
             mcmc_real *dat,    /* [ndat] */
             mcmc_real *std2r,  /* [ndat] */
             mcmc_int  *icoor,  /* [nicoor] */
             mcmc_int  *idat,   /* [nidat] */
             mcmc_real *datm,   /* [ndatm] */
             mcmc_real *chi2m,  /* [nbeta][nseq][ndat] */
             /* Parameters */
             mcmc_real *pcur,   /* [nbeta][nseq][nptot] */
             mcmc_int  *flag,   /* [nbeta][nseq] */
             mcmc_real *chi2c); /* [nbeta][nseq] */

  ~CCalcModel();

  void do_model_chi2_gpu();
  void do_calc_model_gpu();
};



/*====================================================================*/
/*                        DERIVED CLASS CMcmcFit                      */




class CMcmcFit: public CCalcModel {
 public:

  int nprm;
  int nadj;
  int npass;
  int nburn;
  int niter;
  int nrndst;
  int ipm;  /* Loop parameter: current model parameter */
  int itr;  /* Loop parameter: current iteration */
  unsigned long long seed;
  int ntitr;  /* Total number of iterations */

  /*
   * Array sizes (in bytes).
   */
  int s_rndst;  /* = sizeof(curandState)*nrndst; */
  int s_ptot;   /* = sizeof(mcmc_real)*nptot; */
  int s_prm;    /* = sizeof(mcmc_int)*nprm; */
  int s_pout;   /* = sizeof(mcmc_real)*nprm*nbeta*nseq*niter; */
  int s_tout;   /* = sizeof(mcmc_int)*nbeta*nseq*niter; */
  int s_chi2;   /* = sizeof(mcmc_real)*nseq*niter; */
  int s_beta;   /* = sizeof(mcmc_real)*nbeta; */
  int s_acpt;   /* = sizeof(mcmc_int)*nprm*nbeta*nseq; */
  int s_hist;   /* = sizeof(mcmc_int)*nadj*nprm*nbeta*nseq; */
  int s_bsptot; /* = sizeof(mcmc_real)*nbeta*nseq*nptot; */
  int s_4bstf;  /* = sizeof(mcmc_real)*nbeta*nseq*ntitr; */
  int s_bstf;   /* = sizeof(mcmc_real)*nbeta*nseq*ntitr; */
  int s_bsti;   /* = sizeof(mcmc_int)*nbeta*nseq*ntitr; */
  int s_burn;   /* = sizeof(mcmc_real)*nprm*nbeta*nseq*nburn; */

  /*
   * DEVICE pointers to data.
   */
  /* Random Number Generators' States */
  curandState *rndst;     /* [nbeta][nseq], always 48 bytes each */
  /* Parameters */
  mcmc_int  *pdescr;   /* [nptot] */
  mcmc_int  *ivar;     /* [nprm] */
  mcmc_int  *invar;    /* [nptot] */
  mcmc_real *ptotal;   /* [nptot] */
  mcmc_real *pmint;    /* [nptot] */
  mcmc_real *pmaxt;    /* [nptot] */
  /* MCMC Algorithm */
  mcmc_real *pout;     /* [nprm][nbeta][niter,nseq] */
  mcmc_int  *tout;     /* [nbeta][nseq][niter] */
  mcmc_real *chi2;     /* [nseq][niter] */
  mcmc_int  *tcur;     /* [nbeta][nseq] */
  mcmc_int  *n_acpt;   /* [nprm][nbeta][nseq] */
  mcmc_int  *n_exch;   /* [nbeta][nseq] */
  mcmc_int  *n_cnt;    /* [nprm][nbeta][nseq] */
  mcmc_int  *n_hist;   /* [nadj][nprm][nbeta][nseq] */
  mcmc_real *pstp;     /* [nprm][nbeta][nseq] */
  mcmc_real *beta;     /* [nbeta] */
  mcmc_real *ptent;    /* [nbeta][nseq] */
  mcmc_int  *ptentn;   /* [nbeta][nseq] */

  /*
   * HOST Pointers to data
   */
  mcmc_int  *pdescr_h;   /* [nptot] */
  mcmc_int  *ivar_h;     /* [nprm] */
  mcmc_int  *invar_h;    /* [nptot] */
  mcmc_real *ptotal_h;   /* [nptot] */
  mcmc_real *pmint_h;    /* [nptot] */
  mcmc_real *pmaxt_h;    /* [nptot] */
  mcmc_real *pout_h;     /* [nprm][nbeta][niter,nseq] */
  mcmc_int  *tout_h;     /* [nbeta][nseq][niter] */
  mcmc_real *chi2_h;     /* [nseq][niter] */
  mcmc_int  *tcur_h;     /* [nbeta][nseq] */
  mcmc_int  *n_acpt_h;   /* [nprm][nbeta][nseq] */
  mcmc_int  *n_exch_h;   /* [nbeta][nseq] */
  mcmc_int  *n_cnt_h;    /* [nprm][nbeta][nseq] */
  mcmc_int  *n_hist_h;   /* [nadj][nprm][nbeta][nseq] */
  mcmc_real *pstp_h;     /* [nprm][nbeta][nseq] */
  mcmc_real *beta_h;     /* [nbeta] */
  mcmc_real *ptent_h;    /* [nbeta][nseq] */
  mcmc_int  *ptentn_h;   /* [nbeta][nseq] */
  curandState *rndst_h;  /* [nbeta][nseq] */


/*=========================================================================*/


  CMcmcFit(int imodel,
           int ncoor, int ndat, int nicoor, int nidat, int ndatm,
           int nptot, int nprm, int nadj, int npass,
           int nbeta, int nseq, int nburn, int niter,
           /* Random Number Generators */
           unsigned long long seed,
           mcmc_rng *rndst,
           /* Coordinates, Observables and Model Data, Integer Data */
           mcmc_real *coor,   /* [ncoor] */
           mcmc_real *dat,    /* [ndat] */
           mcmc_real *std2r,  /* [ndat] */
           mcmc_int  *icoor,  /* [nicoor] */
           mcmc_int  *idat,   /* [nidat] */
           mcmc_real *datm,   /* [ndatm] */
           mcmc_real *chi2m,  /* [nbeta][nseq][ndat] */
           /* Parameters */
           mcmc_int  *pdescr, /* [nptot] */
           mcmc_int  *ivar,   /* [nprm] */
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
           mcmc_real *pcur);  /* [nbeta][nseq][nptot] */


  ~CMcmcFit();

  void do_mcmc_on_gpu();

};


/*====================================================================*/


void run_mcmc(CMcmcFit *mc_h, CMcmcFit *mc_d);
void run_calcmodchi2(CCalcModel *mc_h, CCalcModel *mc_d);
void run_calcmodel(CCalcModel *mc_h, CCalcModel *mc_d);
void cudaAssert(const cudaError err, const char *file, const int line);

#endif  /* __cplusplus */

#endif  /* MCMCJOB_CUH */
