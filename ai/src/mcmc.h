/* mcmc.h */
/* sizeof(curandState) = 48 B */

#ifndef MCMC_H
#define MCMC_H

#include "mcmc_types.h"

#ifdef __cplusplus
extern "C" {
#endif

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
               int nbeta, int nseq, int nburn, int niter);


void calc_modchi2(mcmc_real *coor, mcmc_real *dat, mcmc_real *std2r,
                  mcmc_int *icoor, mcmc_int *idat, mcmc_real *datm,
                  mcmc_real *chi2m,
                  mcmc_real *pcur, mcmc_real *chi2c, mcmc_int *flag,
                  int imodel,
                  int ncoor, int ndat, int nicoor, int nidat, int ndatm,
                  int nptot, int nbeta, int nseq);


void calc_model(mcmc_real *coor, mcmc_int *idat, mcmc_real *datm,
                mcmc_real *pcur,
                int imodel, int ncoor, int ndat, int nidat,
                int ndatm, int nptot, int nbeta, int nseq);


#ifdef __cplusplus
};
#endif

#endif /* MCMC_H */
