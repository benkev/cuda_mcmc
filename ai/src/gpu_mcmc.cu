//
// gpu_mcmc.cu
//
// CUDA kernels for the MCMC with replica exchange (MCMC-RE).
// Types: mcmc_real for critical arrays (dat, datm, chi2m, chi2c, chi2,
// pcur, pout, ptent, pstp, beta, std2r, coor, ptotal, pmint, pmaxt),
// mcmc_int for indices/counters/flags, mcmc_rng for RNG state.
// Switch float/double with -DMCMC_DOUBLE.
//
// The MCMC algorithm is typically used for fitting parameterized models
// to large arrays of data located at many coordinate points. For example,
// it can be a model of celestial radio object observed with a radio
// interferometer. The data here are the visibilities, and the coordinates are
// u and v, the locations of visibilities in the spatial frequency domain.
// The data should be provided to the algorithm in the dat[ndat] array, and
// the coordinates -- in the coor[ncoor] array. Sometimes the model requires
// integer data, passed in the integer array idat[nidat], and integer
// coordinates (like antenna numbers) are passed in the icoor[nicoor] array.
// To compute the chi^2 the algorithm requires the variance std^2.
// The std2r[ndat] array should contain the reciprocals of the data variances.
//
// -----------------------------------------------------------------------------
//
// See mcmcjob.cuh for the class definitions and mcmc_types.h for the types.
//
//

#include <cuda.h>
#include <curand_kernel.h>
#include <stdio.h>
#include "mcmcjob.cuh"
#include "model.cuh"

#define Nbeta (blockDim.x)
#define Nseq (gridDim.x)
#define Ibeta (threadIdx.x)
#define Iseq (blockIdx.x)


/*
 * Initialize states one for all the threads
 */
__global__ void init_rng(CMcmcFit *mc) {

  int ist = threadIdx.x*gridDim.x + blockIdx.x;

  if (ist >= mc->nrndst) return;

  unsigned long long offset = 0;

  curand_init(mc->seed, ist, offset, &mc->rndst[ist]);
}


//==========================================================
/* init_pcur<<<nseq,nbeta>>>(mc_d); */

__global__ void init_pcur(CMcmcFit *mc) {

  int ist = Nseq*Ibeta + Iseq;    /* [ibeta,iseq] */
  int k, i;
  mcmc_real p, randu;

  if (ist >= mc->nrndst) return;

  mc->flag[ist] = 1; /* All parameters initially are accepted */
  mc->itr = 0;

  for (i = 0; i < mc->nprm; i++) {
    k = mc->ivar[i];  /* turn i in [0..nprm-1] into k in [0..nptot-1] */

    /* Set p to a urandom value in [pmint[k]..pmaxt[k]] */
    randu = (mcmc_real)curand_uniform(&mc->rndst[ist]);
    p  = mc->pmint[k] + (mc->pmaxt[k] - mc->pmint[k])*randu;

    /* If p is angle, correct angle into +-pi */
    if (mc->pdescr[k] == 2) p = (mcmc_real)atan2(sin((double)p),cos((double)p));
    mc->pcur[ist*mc->nptot+k] = p; /* pcur[ibeta,iseq,k] = p */
  }
}

//==========================================================

/*     calc_chi2_terms<<<nblocks,blocksize>>>(mc_d, ipass);  */

__global__ void calc_chi2_terms(CCalcModel *mc, int ipass) {

  int idatm = blockDim.x*blockIdx.x + threadIdx.x; /* through index */

  if (idatm >= mc->nchi2m) return; /* The last block can have extra threads */

  int ist, ipt, idat, iret;
  mcmc_real dif;

  /*
   * ndat = nvis + ncph
   *
   * Treat the through index idatm as idatm ~ [ibeta,iseq,idat], or
   * idatm = (ibeta*nseq + iseq)*ndat + idat:
   */
  idat = idatm%mc->ndat;
  ist =  idatm/mc->ndat;      /* ptent[ist] ~ ptent[ibeta,iseq] */

  if (mc->flag[ist] == 0) return; // ==== no calcs for rejected parameters ===>>

  ipt = ist*mc->nptot;     /* pcur[ipt] ~ pcur[ibeta,iseq,:] */

  /*
   * Call model
   */

  iret = model(mc, idat, ipt, idatm, ipass); /* Compute model at idatm */

  if (iret == 0) return; /* No results */
  /*
   * Calculate one term of the chi^2 sum
   */
  dif = mc->dat[idat] - mc->datm[idatm];

  if (iret == 2) { /* I.e. dif is closure phase difference */
    dif = (mcmc_real)atan2(sin((double)dif), cos((double)dif));
  }

  mc->chi2m[idatm] = (dif*dif)*mc->std2r[idat];   /* (xi-Xi)^2*(1/std^2) */

}

//========================================================

// calc_chi2<<<nseq,nbeta>>>(mc_d);

__global__ void calc_chi2(CCalcModel *mc) {
  /*  */
  int ist = Nseq*Ibeta + Iseq;     /* [ibeta,iseq] */

  int ier0 = ist*mc->ndat;         /* [ibeta,iseq,0] */
  int ier1 = ier0 + mc->ndat;      /* [ibeta,iseq,ndat-1] */
  int i;
  mcmc_real sum = (mcmc_real)0.0;

  /*
   * If the parameter set was rejected (flag==0), chi2m[] contains
   * stale values. Mark chi2c[ist] as invalid with +Inf instead of
   * summing stale terms.
   */
  if (mc->flag[ist] == 0) {
    mc->chi2c[ist] = (mcmc_real)INFINITY;
    return;
  }

  for (i = ier0; i < ier1; i++) {
    sum += mc->chi2m[i];
  }
  mc->chi2c[ist] = sum;

  /* Now chi2c[nbeta,nseq] contains the new chi^2 for all temperatures and
   * all sequences. */
}

//==========================================================

/*   gen_proposal<<<nseq,nbeta>>>(mc_d); */

__global__ void gen_proposal(CMcmcFit *mc) {

  int ist = Nseq*Ibeta + Iseq;    /* [ibeta,iseq] */
  int k, t, pt;
  mcmc_real r, p, std, randn;

  if (ist >= mc->nrndst) return;

  /* Randomly select parameter number k to make step */
  r = (mcmc_real)curand_uniform(&mc->rndst[ist]);
  k = (int) (mc->nprm*r); /* Random parameter index 0..nprm-1 into ivar[] */
  if (k >= mc->nprm) k = mc->nprm - 1;  /* Guard against r == 1.0 */
  t = mc->ivar[k]; /* Random parameter index into pcur[ibeta,iseq,:]  */
  mc->ptentn[ist] = t; /* The param # in pcur chosen for variation */

  /* Select a t'th parameter (t is random!) from pcur[ibeta,iseq,t] */
  pt = ist*mc->nptot + t; /* Index into pcur[ibeta,iseq,t] */
  p = mc->pcur[pt]; /* p = pcur[ibeta,iseq,t] */


  /* Save the current parameter value in ptent[ibeta,iseq] */
  mc->ptent[ist] = p;               /* ptent[ibeta,iseq] = p; */

  /* Add to it a "Gaussian step" - noise with stdev = pstp[k,ibeta,iseq] */
  std = mc->pstp[(Nbeta*k + Ibeta)*Nseq + Iseq]; /* pstp[k,ibeta,iseq] */
  randn = (mcmc_real)curand_normal(&mc->rndst[ist]);
  p = p + std*randn;

  if (mc->pdescr[t] == 2)
    p = (mcmc_real)atan2(sin((double)p),cos((double)p)); /* Correct angle into +-pi */

  /* check_p(); */
  if ((p < mc->pmint[t]) || (p > mc->pmaxt[t])) {/* If outside of the prior: */
    mc->flag[ist] = 0;  /* Flag the parameter as 'outside of the prior' */
                        /* and leave pcur[ibeta,iseq,:] intact */
  }
  else {
    mc->flag[ist] = 1;  /* Flag the parameter as 'inside of the prior' */
    /* Save the new parameter in pcur[ibeta,iseq,t]*/
    mc->pcur[pt] = p;  /* pcur[ibeta,iseq,t] = p; - put new parameter value */

  }
  /* Now:
   * ptentn[nbeta,nseq] has the numbers (0:nptot-1) of the moved parameters.
   * If a parameter is out of the prior, flag[ibeta,iseq] is set to 0,
   * and pcur[ibeta,iseq,t] is left with the old parameter value.
   * Otherwise flag[ibeta,iseq] is set to 1, and
   * pcur[ibeta,iseq,t] has the variated parameter.
   */

}  /* __global__ void gen_proposal(CMcmcFit *mc)  */


//==========================================================
//
// Single Parameter Metropolis MCMC
//
// spmp_mcmc<<<nseq,nbeta>>>(mc_d);

__global__ void spmp_mcmc(CMcmcFit *mc) {
  /*  */
  int ist = Nseq*Ibeta + Iseq;      /* [ibeta,iseq] */

  if (ist >= mc->nrndst) return;

  if (mc->flag[ist] == 0) return;   /* 0: param was outside of prior */

  int itrm0 = ist*mc->ndat;         /* chi2m[ibeta,iseq,0] */
  int itrm1 = itrm0 + mc->ndat;     /* chi2m[ibeta,iseq,ndat-1] */
  int i, ip;
  mcmc_real u, a, alpha, chi2 = (mcmc_real)0.0;
  /* Here k is the parameter number randomly chosen in gen_proposal() */
  int k = mc->invar[mc->ptentn[ist]]; /* [0:nprm-1] from [0:nptot-1] */
  int iacpt;   /* iacpt ~ [k,ibeta,iseq] */

  /* Here ptent[ibeta,iseq] parameter is within the prior */
  /* Find chi^2 for model data at (ibeta,iseq) */
  for (i = itrm0; i < itrm1; i++)
    chi2 += mc->chi2m[i];

  a = (mcmc_real)exp(-(mcmc_real)0.5*mc->beta[Ibeta]*(chi2 - mc->chi2c[ist]));
  alpha = min(a, (mcmc_real)1);
  u = (mcmc_real)curand_uniform(&mc->rndst[ist]);

  if (u <= alpha) { /* Test OK? */
    mc->chi2c[ist] = chi2; /* chi2c[ibeta,iseq] = chi2; new prm is in pcur */

    /* n_acpt[nprm,nbeta,nseq]:  n_acpt[k,ibeta,inseq]++; */
    iacpt = (Nbeta*k + Ibeta)*Nseq + Iseq; /* iacpt ~ [k,ibeta,iseq] */
    mc->n_acpt[iacpt]++;  /* Accepted! n_acpt[k,ibeta,iseq]++; */
  }
  else {/* Test failed? */
    /* pcur[ibeta,iseq,ptentn[ibeta,iseq]] = ptent[ibeta,iseq] */
    ip = ist*mc->nptot + mc->ptentn[ist];

    mc->pcur[ip] = mc->ptent[ist]; /* Revert pcur */
    mc->flag[ist] = 0; /* Flag the parameter as 'did not pass chi2 test */
  }
}

//==========================================================

// adjust_steps<<<nseq,nbeta>>>(mc_d);

__global__ void adjust_steps(CMcmcFit *mc) {
  /*  */
  int ist = Nseq*Ibeta + Iseq;    /* [ibeta,iseq] */

  if (ist >= mc->nrndst) return;

  int modn, ihist, sumh;
  int iadj;
  mcmc_real rate;
  /* Here k is the parameter number randomly chosen in gen_proposal() */
  int k = mc->invar[mc->ptentn[ist]]; /* [0:nprm-1] from [0:nptot-1] */
  int ip = (Nbeta*k + Ibeta)*Nseq + Iseq; /* [k,ibeta,iseq] */

  /* modn - index into n_hist[modn,k,ibeta,iseq] */
  modn = mc->n_cnt[ip] % mc->nadj;  /* modn = n_cnt[k,ibeta,iseq] % nadj; */

  /* n_hist[nadj,nprm,nbeta,nseq]; ihist ~ [modn,k,ibeta,iseq]. */
  ihist = ((mc->nprm*modn + k)*Nbeta + Ibeta)*Nseq + Iseq;


  /*  Memorize the history of accept/reject */
  if (mc->flag[ist] != 0)
    mc->n_hist[ihist] = 1; /* Accepted: n_hist[modn,k,ibeta,iseq] = 1 */
  else
    mc->n_hist[ihist] = 0; /* Rejected: n_hist[modn,k,ibeta,iseq] = 0 */


  /* Calculate accept rate of the latest nadj trials */
  sumh = 0;
  for (iadj = 0; iadj < mc->nadj; iadj++) {
    /* ihist ~ [iadj,k,ibeta,iseq]. */
    ihist = ((mc->nprm*iadj + k)*Nbeta + Ibeta)*Nseq + Iseq;
    sumh += mc->n_hist[ihist];
  }

  rate = ((mcmc_real) sumh) / ((mcmc_real) min(mc->nadj,mc->n_cnt[ip]));

  /* Adjust the step size: >30% -> x1.01; <20% -> /1.01; otherwise unchanged */
  if (rate > (mcmc_real)0.30)
    mc->pstp[ip] *= (mcmc_real)1.01;
  else if (rate < (mcmc_real)0.20)
    mc->pstp[ip] /= (mcmc_real)1.01;
  /* else: do not change */

  mc->n_cnt[ip] += 1;

}


//==========================================================

/* replica_exchange<<<nseq,1>>>(mc_d, burn, itr); */

__global__ void replica_exchange(CMcmcFit *mc, int burn, int itr) {
  /*  */
  int ist = Iseq;
  int nbeta_minus_1, idum, k0, k1, i0, i1;
  mcmc_real u, alpha;
  mcmc_real dum, r;
  int ipo, ipc, ico, k, jprm, ibeta, ipm, ito, itc;


  /* Do replica-exchange - can be done only in one thread in each block*/
  if (Ibeta != 0) return;

  nbeta_minus_1 = mc->nbeta - 1;

  for (ibeta = 0; ibeta < mc->nbeta; ibeta++) {
    /* Randomly select adjacent chains */
    r = (mcmc_real)curand_uniform(&mc->rndst[ist]);
    k = (int) (nbeta_minus_1*r);
    if (k >= nbeta_minus_1) k = nbeta_minus_1 - 1;  /* Guard r==1.0 */

    /* calculate metropolis ratio in selected adjacent temperature chains */
    k0 = k*Nseq + Iseq;       /* [k,nsec] in [nbeta,nsec] */
    k1 = (k+1)*Nseq + Iseq;   /* [k+1,nsec] in [nbeta,nsec] */
    r = (mcmc_real)exp(-(mcmc_real)0.5*(mc->beta[k+1] - mc->beta[k]) * \
	    (mc->chi2c[k0] - mc->chi2c[k1]));

    /* calculate probability for acceptance */
    alpha = min(r, (mcmc_real)1);
    /* exchange adjacent temperature chains with a probability alpha */
    u = (mcmc_real)curand_uniform(&mc->rndst[ist]);

    if (u <= alpha) {
      /* exchange tempertures */
      idum = mc->tcur[k0];
      mc->tcur[k0] = mc->tcur[k1];
      mc->tcur[k1] = idum;

      /* exchange parameters pcur[k,iseq,:] and pcur[k+1,iseq,:] */
      for(jprm = 0; jprm < mc->nptot; jprm++) {
          i0 = k0*mc->nptot + jprm;
          i1 = k1*mc->nptot + jprm;
          dum = mc->pcur[i0];
          mc->pcur[i0] =  mc->pcur[i1];
          mc->pcur[i1] = dum;
      } /* for(jprm = 0; jprm < mc->nptot; jprm++) */

      /* exchange chisquares */
      dum = mc->chi2c[k0];
      mc->chi2c[k0] = mc->chi2c[k1];
      mc->chi2c[k1] = dum;

      /* At the main stage, n_exch[k,iseq]++; */
      if (burn == 0) mc->n_exch[k*Nseq+Iseq]++;


    } /* if (u <= alpha)  */
  } /* for (i = 0; i < Nbeta; i++)  */

 /*
  * This is the last iteration point.
  * Increment the internal total iteration counter:
  */
  if (Iseq == 0) mc->itr++;

  if (burn) return; /* At the burn-in stage pout[] is not updated */

  /*
   * Saving results
   */
  /* Save pcur in pout[itr], tcur in tout[itr], and chi2c in chi2 */
  for (ibeta = 0; ibeta < mc->nbeta; ibeta++) {
    for (ipm = 0; ipm < mc->nptot; ipm++) {
      if (mc->invar[ipm] != -1) {
        ipo = ((mc->nbeta*mc->invar[ipm] + ibeta)*Nseq + Iseq)*mc->niter + itr;
        ipc = (Nseq*ibeta + Iseq)*mc->nptot + ipm;
        mc->pout[ipo] = mc->pcur[ipc];
      }
    } /* for (ipm = 0; ipm < mc->nptot; ipm++) */
    /* tout[ibeta,iseq,itr] = tcur[ibeta,iseq] */
    ito = (Nseq*ibeta + Iseq)*mc->niter + itr;
    itc = Nseq*ibeta + Iseq;
    mc->tout[ito] = mc->tcur[itc];
  } /* for (ibeta = 0; ibeta < mc->nbeta; ibeta++) */

  /* chi2[iseq,itr] = chi2c[0,iseq] */
  ico = mc->niter*Iseq + itr;
  mc->chi2[ico] = mc->chi2c[Iseq];

}


//==========================================================

/*     calc_model<<<nblocks,blocksize>>>(mc_d);  */

__global__ void calc_model(CCalcModel *mc) {

  int idatm = blockDim.x*blockIdx.x + threadIdx.x; /* through index */

  if (idatm >= mc->nchi2m) return; /* The last block can have extra threads */

  int ist, ipt, idat;

  /*
   * ndat = nvis + ncph
   *
   * Treat the through index idatm as idatm ~ [ibeta,iseq,idat], or
   * idatm = (ibeta*nseq + iseq)*ndat + idat:
   */
  idat = idatm%mc->ndat;
  ist =  idatm/mc->ndat;      /* ptent[ist] ~ ptent[ibeta,iseq] */

  if (mc->flag[ist] == 0) return; // ==== no calcs for rejected parameters ===>>

  ipt = ist*mc->nptot;     /* pcur[ipt] ~ pcur[ibeta,iseq,:] */

  /*
   * Call model with ipass=0 (last parameter)
   */
  model(mc, idat, ipt, idatm, 0);  /* Compute model at idatm */
}

#undef Nbeta
#undef Nseq
#undef Ibeta
#undef Iseq

//==========================================================


void run_mcmc(CMcmcFit *mc_h, CMcmcFit *mc_d) {

  int t;
  int nseq = mc_h->nseq, nbeta = mc_h->nbeta;
  int blocksize = 128;
  int nblocks= mc_h->nchi2m/blocksize + 1;     /* nchi2m = nbeta*nseq*ndat */
  int itr, niter = mc_h->niter, ipm, nprm = mc_h->nprm;
  int nburn = mc_h->nburn;
  int burn; /* Flag 1: 'burn-in stage'; 0: 'main stage' */
  /* Pass 0: model() calculates amplitudes and phases;
   * Pass 1: model() calculates closures from precalculated phases */
  int ipass = 0;
  int nptot = mc_h->nptot;

  printf("HOST MCMC_REAL = %s\n", MCMC_REAL_STR);

  /* Loop parameters: */
  mc_h->ipm = 0;
  mc_h->itr = 0;

  printf("HOST imodel = %d\n", mc_h->imodel);
  printf("HOST pmint:\n");
  for (t = 0; t < nptot; t++) printf("%g  ", (double)mc_h->pmint_h[t]);
  printf("\n");
  printf("HOST pmaxt:\n");
  for (t = 0; t < nptot; t++) printf("%g  ", (double)mc_h->pmaxt_h[t]);
  printf("\n");
  printf("HOST ptotal:\n");
  for (t = 0; t < nptot; t++) printf("%g  ", (double)mc_h->ptotal_h[t]);
  printf("\n");

  /*
   * Initialization
   */
  init_rng<<<nseq,nbeta>>>(mc_d);
  init_pcur<<<nseq,nbeta>>>(mc_d);
  for (ipass = 0; ipass < mc_h->npass; ipass++) {
    calc_chi2_terms<<<nblocks,blocksize>>>(mc_d, ipass);
  }
  calc_chi2<<<nseq,nbeta>>>(mc_d);

  printf("HOST nburn = %d, nprm = %d\n", nburn, nprm);

  /* Burn-in stage */
  burn = 1;
  for (itr = 0; itr < nburn; itr++) {
    for (ipm = 0; ipm < nprm; ipm++) {
      gen_proposal<<<nseq,nbeta>>>(mc_d);
      for (ipass = 0; ipass < mc_h->npass; ipass++) {
          calc_chi2_terms<<<nblocks,blocksize>>>(mc_d, ipass);
      }
      spmp_mcmc<<<nseq,nbeta>>>(mc_d);
      adjust_steps<<<nseq,nbeta>>>(mc_d);
    }
    replica_exchange<<<nseq,1>>>(mc_d, burn, itr);
    if (itr%200 == 0 && itr > 0) printf("BURN-IN: itr = %d done\n", itr);
  }
  printf("BURN-IN: total of %d iterations done\n", nburn);

  printf("Burn-in done\n");


  /* Main stage */
  burn = 0;
  for (itr = 0; itr < niter; itr++) {
    for (ipm = 0; ipm < nprm; ipm++) {
      gen_proposal<<<nseq,nbeta>>>(mc_d);
      for (ipass = 0; ipass < mc_h->npass; ipass++) {
          calc_chi2_terms<<<nblocks,blocksize>>>(mc_d, ipass);
      }
      spmp_mcmc<<<nseq,nbeta>>>(mc_d);
    }
    replica_exchange<<<nseq,1>>>(mc_d, burn, itr);

    if (itr%200 == 0 && itr > 0) printf("MCMC: itr = %d done\n", itr);
  }

  printf("MCMC:  total of %d iterations done\n", niter);

  printf("GPU: runmcmc() done.\n");
  return;
}


/*======================================================================*/


void run_calcmodchi2(CCalcModel *mc_h, CCalcModel *mc_d) {

  int blocksize = 128;
  int nblocks= mc_h->nchi2m/blocksize + 1;     /* nchi2m = nbeta*nseq*ndat */
  int nbeta = mc_h->nbeta, nseq = mc_h->nseq;

  printf("nbeta=%d, nseq=%d, nblocks=%d, blocksize=%d, mc_h->nchi2m=%d\n",
	 mc_h->nbeta, mc_h->nseq, nblocks, blocksize, mc_h->nchi2m);

  calc_chi2_terms<<<nblocks,blocksize>>>(mc_d, 0);
  calc_chi2_terms<<<nblocks,blocksize>>>(mc_d, 1);

  calc_chi2<<<nseq,nbeta>>>(mc_d);
  return;
}



/*======================================================================*/


void run_calcmodel(CCalcModel *mc_h, CCalcModel *mc_d) {

  int blocksize = 128;
  int nblocks= mc_h->nchi2m/blocksize + 1;     /* nchi2m = nbeta*nseq*ndat */
  int nbeta = mc_h->nbeta, nseq = mc_h->nseq;

  printf("nbeta=%d, nseq=%d, nblocks=%d, blocksize=%d, mc_h->nchi2m=%d\n",
	 nbeta, nseq, nblocks, blocksize, mc_h->nchi2m);

  calc_model<<<nblocks,blocksize>>>(mc_d);

  return;
}
