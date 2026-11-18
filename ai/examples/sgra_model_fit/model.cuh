//#include <curand_kernel.h>
//#include <stdio.h>
#include "mcmc_types.h"

__device__ void sgra_model9(mcmc_real ul, mcmc_real vl, mcmc_real *prm, 
			   mcmc_real *Vamp, mcmc_real *Vpha);
__device__ void sgra_model13(mcmc_real ul, mcmc_real vl, mcmc_real *prm, 
			   mcmc_real *Vamp, mcmc_real *Vpha);

__device__ int model(CCalcModel *mc, int id, int ipt, int imd, int ipass) {

  /* Arguments:
   *   id: index into mc->dat. 
   *   ipt: index of the first parameter in the set of model parameters 
   *        for ibeta == threadIdx.x and iseq == blockIdx.x. 
   *        mc->pcur[ipt] is the first model parameter, and 
   *        mc->pcur[ipt+mc->nptot-1] is the last parameter.
   *        If i is a parameter index, from 0 to mc->nptot-1,
   *        then mc->pcur[ipt+i] is the i-th parameter, or
   *        pcur[ibeta,iseq,i] in Python.
   *   imd: "through index" into datm[ibeta,isec,id]. Only used 
   *        to save the result before return.
   *   ipass: pass number, starting from 0. The model() function can be
   *          called multiple times, or in many passes, so pass is to know
   *          which time model() is called now.
   *
   * The result must be saved in
   *   mc->datm[imd]
   *
   * All member arrays and parameters of the class CCalcModel are accessible
   * here. Use the expressions like mc->coor, mc->idat, mc->ncoor, 
   * mc->nidat etc. See the class CCalcModel definition in mcmcjob.cuh and
   * mcmcjob.cu.
   * 
   */


  /* Data and coordinates */
  int nvis = mc->idat[0];
  int nchi2m = mc->nchi2m; /* Number of chi2 terms & start of mphase arrays */
  int ivis = imd/mc->ndat; /* Throug-index ph[ivis,:] ~ ph[ibeta,iseq,:] */

  mcmc_real *phase =  mc->datm + nchi2m + ivis*nvis; /* phase[ibeta,iseq,i]*/

  if (ipass == 0 && id < nvis) { /* Amplitude and phase model calculation */
    int iuv = 2*id;
    mcmc_real ul = mc->coor[iuv];
    mcmc_real vl = mc->coor[iuv+1];
    mcmc_real *prm = &mc->pcur[ipt];
    /*
     *   imodel: a model number. Here  imodel = 1 means sgra_model9(), and
     *                                 imodel = 2 means sgra_model13().
     */
    if (mc->imodel == 1) {
      sgra_model9(ul, vl, prm,  &mc->datm[imd], &phase[id]);
    }
    else if (mc->imodel == 2) { 
      sgra_model13(ul, vl, prm,  &mc->datm[imd], &phase[id]);
    }
    else return -1;                 /* Error: wrong model number */

    return 1;
  }


  else if (ipass == 1 && id >= nvis) { /* Closures from saved model phases */
    mcmc_real cph;
    if (mc->idat[2] == 0) { /* cpExt == 0: calc cphs from phases */
      int icph = 3*(id - nvis); /* Phase index triplet number, 0 : 3*(ncph-1) */
      int i = mc->icoor[icph++];
      int j = mc->icoor[icph++];
      int k = mc->icoor[icph];
      phase =  mc->datm + nchi2m + ivis*nvis;
      cph = phase[i] + phase[j] - phase[k];     /* Phase closure */
    }
    else { /* cpExt != 0: cphs and uvcp are provided */
      mcmc_real *prm = &mc->pcur[ipt];
      mcmc_real amp, ph1, ph2, ph3;
      mcmc_real u1, u2, u3, v1, v2, v3;
      int icph = 4*(id - nvis) + 2*nvis; /* Index of [u1,v1,u2,v2] in coor */
      u1 = mc->coor[icph++]; 
      v1 = mc->coor[icph++];
      u2 = mc->coor[icph++];
      v2 = mc->coor[icph];
      u3 = -(u1 + u2);
      v3 = -(v1 + v2);
      if (mc->imodel == 1) {
	sgra_model9(u1, v1, prm,  &amp, &ph1);
	sgra_model9(u2, v2, prm,  &amp, &ph2);
	sgra_model9(u3, v3, prm,  &amp, &ph3);
      }
      else if (mc->imodel == 2) { 
	sgra_model13(u1, v1, prm,  &amp, &ph1);
	sgra_model13(u2, v2, prm,  &amp, &ph2);
	sgra_model13(u3, v3, prm,  &amp, &ph3);
      }
      else return -1;                 /* Error: wrong model number */
      cph = ph1 + ph2 - ph3;
    }
    mc->datm[imd] = atan2(sin(cph), cos(cph));

    return 2;
  }

  else
    return 0;
}

/*========================================================================*/


__device__ void sgra_model13(mcmc_real ul, mcmc_real vl, mcmc_real *prm, 
			     mcmc_real *Vamp, mcmc_real *Vpha) {

  /* Model xringaus2 (13-parameter)
   * Arguments:
   * ul, vl: u and v coordinates in Gigalambda
   * prm: model parameters
   * Result:
   * Vamp: visibility amplitude of the model at (ul,vl) points 
   * Vpha: visibility phase (radians) of the model at (ul,vl) points 
   */

    
  /* Locals */
  mcmc_real Zsp, Rex, rq, ecc, fade, gr, gax, aq, gq, alpha, beta, eta, th;
  mcmc_real D, d, gsy, gsx, h, h0, etth;
  mcmc_real U, V; /* Coordinates rortated by th */
  mcmc_real rho, arg, ree, ime, rei, imi;
  mcmc_real Rin, aRin, aRex;
  mcmc_real gshift, fwhm2std, Gamp, Gph;
  mcmc_real d_re, Fr_re, Fg_re, Fin_re;
  mcmc_real d_im, Fr_im, Fg_im, Fin_im;
  mcmc_real Vim, Vre;

  /* Zsp, Rex, rq, ecc, fade, gr, gax, aq, gq, alpha, beta, eta, th = vparam */
  Zsp = prm[0];
  Rex = prm[1];
  rq = prm[2];
  ecc = prm[3];
  fade = prm[4]; 
  gr = prm[5]; 
  gax = prm[6]; 
  aq  = prm[7];
  gq = prm[8]; 
  alpha = prm[9];
  beta = prm[10];
  eta = prm[11];
  th = prm[12];

  Rex = (pi/(mcmc_real)180.0)*(Rex/(mcmc_real)3600.0)*(mcmc_real)1e3;
  Rin = rq*Rex;        /* 0 < rq < 1 is the Rin/Rex quotient */
  /*
   * First calculate the Gaussian
   */
  if (gax != (mcmc_real)0.0 && aq != (mcmc_real)0.0 && gq != (mcmc_real)0.0) {
    gsy = gax*Rex;   /* nrad. The Gaussian's axis tangential to the ring */
    gsx = aq*gsy;    /* nrad. The Gaussian's radial axis normal to the ring */
    U =  ul*cos(beta) + vl*sin(beta);
    V = -ul*sin(beta) + vl*cos(beta);
    gshift = gr*Rex*(cos(alpha)*ul + sin(alpha)*vl);   /* Gaussian's shift */
    fwhm2std = (mcmc_real)2.0*pow(pi/((mcmc_real)2.0*sqrt((mcmc_real)2.0*log((mcmc_real)2.0))),(mcmc_real)2);
    Gamp = exp(-fwhm2std*(pow(U*gsx,(mcmc_real)2) + pow(V*gsy,(mcmc_real)2)));
    Gph = -(mcmc_real)2.0*pi*gshift;
    Fg_re = Gamp*cos(Gph);
    Fg_im = Gamp*sin(Gph);
  }
  if (gq == (mcmc_real)1.0) {   /* Gaussian only */
    Vre = Zsp*Fg_re;
    Vim = Zsp*Fg_im;    
    *Vamp = sqrt(Vre*Vre + Vim*Vim);
    *Vpha = atan2(Vim, Vre);
    return;                     //  ==================================>>>
  }

  D = ecc*(Rex - Rin); /* nrad, displacement of inner circle center wrt outer */
  etth = eta - th;
  d =  D*cos(etth);
  h = ((mcmc_real)2.0/pi)/((pow(Rex,(mcmc_real)2) - pow(Rin,(mcmc_real)2)*((mcmc_real)1.0 + d/Rex))*fade
	       + (pow(Rex,(mcmc_real)2) - pow(Rin,(mcmc_real)2)*((mcmc_real)1.0 - d/Rex)));
  h0 = fade*h;

  U =  ul*cos(th) + vl*sin(th);
  V = -ul*sin(th) + vl*cos(th);

  rho = sqrt(pow(U,(mcmc_real)2) + pow(V,(mcmc_real)2));  /* UV-space baselines in Glam */

  if (rho > (mcmc_real)1e-7) {
    arg = (mcmc_real)2.0*pi*rho;   /* in Glambdas */
    aRex = arg*Rex;

    ree = (mcmc_real)0.5*(h+h0)*Rex*j1(aRex)/rho;
    ime = (h-h0)/pi4*(pi*Rex*(j0(aRex) - jn(2,aRex))/pow(rho,(mcmc_real)2) 
		      - j1(aRex)/pow(rho,(mcmc_real)3))*U;
    if (Rin != (mcmc_real)0.0) {
      aRin = arg*Rin;
      rei = (mcmc_real)0.5*(h+h0 - d*(h-h0)/Rex)*Rin*j1(aRin)/rho;
      imi = (h-h0)/pi4*rq*(pi*Rin*(j0(aRin) - jn(2,aRin))/pow(rho,(mcmc_real)2) 
			 - j1(aRin)/pow(rho,(mcmc_real)3))*U;
    }
  }
  else {
    ree = (mcmc_real)0.5*(h+h0)*pi*pow(Rex,(mcmc_real)2);
    ime = -(mcmc_real)0.25*pow(pi,(mcmc_real)2)*(h-h0)*pow(Rex,(mcmc_real)3)*U;
    if (Rin != (mcmc_real)0.0) {
      rei = (mcmc_real)0.5*(h+h0 - d*(h-h0)/Rex)*pi*pow(Rin,(mcmc_real)2);
      imi = -(mcmc_real)0.25*pow(pi,(mcmc_real)2)*(h-h0)*(pow(Rin,(mcmc_real)4)/Rex)*U;
    }
  }

  if (Rin != (mcmc_real)0.0) {
    d_re = cos((mcmc_real)2.0*pi*d*U); /* Displacement factor for inner ring - real */
    d_im = sin((mcmc_real)2.0*pi*d*U); /* Displacement factor for inner ring - imag */
    Fin_re = rei*d_re - imi*d_im;
    Fin_im = rei*d_im + imi*d_re;
    Fr_re = ree - Fin_re;
    Fr_im = ime - Fin_im; 
  }
  else {
    Fr_re = ree;
    Fr_im = ime; 
  }
  /*
   * Shifted Gaussian, if specified
   */
  if (gax == (mcmc_real)0.0 || aq == (mcmc_real)0.0 || gq == (mcmc_real)0.0) { /* Gaussian not specified */
    Vre = Zsp*Fr_re;
    Vim = Zsp*Fr_im;
  }
  else {
    Vre = Zsp*(((mcmc_real)1.0-gq)*Fr_re + gq*Fg_re);
    Vim = Zsp*(((mcmc_real)1.0-gq)*Fr_im + gq*Fg_im);    
  }

  *Vamp = sqrt(Vre*Vre + Vim*Vim);
  *Vpha = atan2(Vim, Vre);
}


 
/*========================================================================*/


__device__ void sgra_model9(mcmc_real ul, mcmc_real vl, mcmc_real *prm, 
			     mcmc_real *Vamp, mcmc_real *Vpha) {

  /* Model xringaus (9-parameter)
   * Arguments:
   * ul, vl: u and v coordinates in Gigalambda
   * prm: model parameters
   * Result:
   * Vamp: visibility amplitude of the model at (ul,vl) points 
   * Vpha: visibility phase (radians) of the model at (ul,vl) points 
   */

    
  /* Locals */
  mcmc_real Zsp, Rex, rq, ecc, fade, gax, aq, gq, th;
  mcmc_real d, gsy, gsx, h, h0;
  mcmc_real U, V; /* Coordinates rortated by th */
  mcmc_real rho, arg, ree, ime, rei, imi;
  mcmc_real Rin, aRin, aRex;
  mcmc_real gshift, gcoef, Gamp, Gph;
  mcmc_real d_re, Fr_re, Fg_re, Fin_re;
  mcmc_real d_im, Fr_im, Fg_im, Fin_im;
  mcmc_real Vim, Vre;

  /* Zsp, Re, rq, ecc, fade, gax, aq, gq, th = vparam */
  Zsp = prm[0];
  Rex = prm[1];
  rq = prm[2];
  ecc = prm[3];
  fade = prm[4]; 
  gax = prm[5]; 
  aq  = prm[6];
  gq = prm[7]; 
  th = prm[8];

  Rex = (pi/(mcmc_real)180.0)*(Rex/(mcmc_real)3600.0)*(mcmc_real)1e3;
  Rin = rq*Rex;        /* 0 < rq < 1 is the Rin/Rex quotient */
  d = ecc*(Rex - Rin); /* nrad, displacement of inner circle center wrt outer */
  gsy = gax*Rex;       /* nrad. The Gaussian's axis tangential to the ring */
  gsx = aq*gsy;       /* nrad. The Gaussian's radial axis normal to the ring */
  /* Convert FWHM^2 to stdev^2 */
  gcoef = (mcmc_real)2.0*pow(pi/((mcmc_real)2.0*sqrt((mcmc_real)2.0*log((mcmc_real)2.0))),(mcmc_real)2);
  h = ((mcmc_real)2.0/pi)/((pow(Rex,(mcmc_real)2) - pow(Rin,(mcmc_real)2)*((mcmc_real)1.0 + d/Rex))*fade
	       + (pow(Rex,(mcmc_real)2) - pow(Rin,(mcmc_real)2)*((mcmc_real)1.0 - d/Rex)));
  h0 = fade*h;

  /*
   * Rotate coordinates by th angle
   */
  U =  ul*cos(th) + vl*sin(th);
  V = -ul*sin(th) + vl*cos(th);

  rho = sqrt(pow(U,(mcmc_real)2) + pow(V,(mcmc_real)2));  /* UV-space baselines in Glam */

  if (rho > (mcmc_real)1e-7) {
    arg = (mcmc_real)2.0*pi*rho;   /* in Glambdas */
    aRex = arg*Rex;

    ree = (mcmc_real)0.5*(h+h0)*Rex*j1(aRex)/rho;
    ime = (h-h0)/pi4*(pi*Rex*(j0(aRex) - jn(2,aRex))/pow(rho,(mcmc_real)2) 
		      - j1(aRex)/pow(rho,(mcmc_real)3))*U;
    if (Rin != (mcmc_real)0.0) {
      aRin = arg*Rin;
      rei = (mcmc_real)0.5*(h+h0 - d*(h-h0)/Rex)*Rin*j1(aRin)/rho;
      imi = (h-h0)/pi4*rq*(pi*Rin*(j0(aRin) - jn(2,aRin))/pow(rho,(mcmc_real)2) 
			 - j1(aRin)/pow(rho,(mcmc_real)3))*U;
    }
  }
  else {
    ree = (mcmc_real)0.5*(h+h0)*pi*pow(Rex,(mcmc_real)2);
    ime = -(mcmc_real)0.25*pow(pi,(mcmc_real)2)*(h-h0)*pow(Rex,(mcmc_real)3)*U;
    if (Rin != (mcmc_real)0.0) {
      rei = (mcmc_real)0.5*(h+h0 - d*(h-h0)/Rex)*pi*pow(Rin,(mcmc_real)2);
      imi = -(mcmc_real)0.25*pow(pi,(mcmc_real)2)*(h-h0)*(pow(Rin,(mcmc_real)4)/Rex)*U;
    }
  }

  if (Rin != (mcmc_real)0.0) {
    d_re = cos((mcmc_real)2.0*pi*d*U);
    d_im = sin((mcmc_real)2.0*pi*d*U);
    Fin_re = rei*d_re - imi*d_im;
    Fin_im = rei*d_im + imi*d_re;
    Fr_re = ree - Fin_re;
    Fr_im = ime - Fin_im; 
  }
  else {
    Fr_re = ree;
    Fr_im = ime; 
  }
  if (gax == (mcmc_real)0.0 || aq == (mcmc_real)0.0 || gq == (mcmc_real)0.0) {
    Vre = Zsp*Fr_re;
    Vim = Zsp*Fr_im;
  }
  else {
    gshift = Rin - d; /* Gaussian centered at the inner edge of inner ring */
    Gamp = exp(-gcoef*(pow(U*gsx,(mcmc_real)2) + pow(V*gsy,(mcmc_real)2)));
    Gph = -(mcmc_real)2.0*pi*gshift*U;
    Fg_re = Gamp*cos(Gph);
    Fg_im = Gamp*sin(Gph);
    Vre = Zsp*(((mcmc_real)1.0-gq)*Fr_re + gq*Fg_re);
    Vim = Zsp*(((mcmc_real)1.0-gq)*Fr_im + gq*Fg_im);    
  }
  *Vamp = sqrt(Vre*Vre + Vim*Vim);
  *Vpha = atan2(Vim, Vre);
}