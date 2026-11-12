//
// model.cuh
//
// The result - a single number - is stored in datm[imd].
//

#include <cuda_runtime.h>
#include "mcmc_types.h"

__device__ mcmc_real planck(mcmc_real T, mcmc_real nu);
__device__ mcmc_real y1_func(mcmc_real theta0, mcmc_real lambda1, mcmc_real T0,
                         mcmc_real psi, mcmc_real nu);
__device__ mcmc_real Sv_func(mcmc_real theta0, mcmc_real lambda1, mcmc_real T0,
                             mcmc_real psi, mcmc_real r0, mcmc_real ymax,
                             mcmc_real nu, mcmc_real d);



static constexpr mcmc_real ak = (mcmc_real)0.212;
static constexpr mcmc_real myu = (mcmc_real)1.67e-24; // * u.g
static constexpr mcmc_real h = (mcmc_real)6.626e-27; // erg-sec, Planck const.
static constexpr mcmc_real k = (mcmc_real)1.38e-16; // erg/deg, Boltzmann const.
static constexpr mcmc_real c = (mcmc_real)3e10;     // cm/sec,  Speed of light
static constexpr mcmc_real c2 = (mcmc_real)9e20; // cm^2/sec^2, (Sp. of light)^2
static constexpr mcmc_real dist = (mcmc_real)5.49e21;
static constexpr mcmc_real c_one = (mcmc_real)1.0;
static constexpr mcmc_real c_two = (mcmc_real)2.0;
static constexpr mcmc_real c_1third = (mcmc_real)0.3333333333333333;

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
   * All member arrays and parameters of the class CCalcModel are accessible
   * here. Use the expressions like mc->coor, mc->idat, mc->ncoor,
   * mc->nidat etc. See the class CCalcModel definition in mcmcjob.cuh and
   * mcmcjob.cu.
   *
   * The result must be saved in
   *   mc->datm[imd]
   */

    mcmc_real nu, Sv;
    mcmc_real theta0 = mc->pcur[ipt++];
    mcmc_real lambda1 = mc->pcur[ipt++];
    mcmc_real T0 =      mc->pcur[ipt++];
    mcmc_real psi =     mc->pcur[ipt++];
    mcmc_real r0 =      mc->pcur[ipt++];
    mcmc_real ymax =    mc->pcur[ipt];

    nu = mc->coor[id];    // A frequency

    Sv = Sv_func(theta0, lambda1, T0, psi, r0, ymax, nu, dist);

    // mcmc_real spect = mc->dat[id];    // A spectrum

    mc->datm[imd] = Sv;

    if (imd < 100)
        printf("\nid=%d, ipt=%d, imd=%d, Sv=%g\n" \
               "theta0=%g, lambda1=%g, T0=%g, psi=%g, r0=%g, ymax=%g, nu=%g, " \
               "dist=%g\n", id, ipt, imd, (double)Sv,
               (double)theta0, (double)lambda1, (double)T0,
               (double)psi, (double)r0, (double)ymax, (double)nu,
               (double)dist);

   return 1;
}

// Bv - распределение Планка
// Spectral radiance of a body for frequency ν at absolute temperature T
__device__ mcmc_real planck(mcmc_real T, mcmc_real nu) {
        return c_two*h*nu*nu*nu/c2/(exp(h*nu/k/T) - c_one);
}


__device__ mcmc_real y1_func(mcmc_real theta0, mcmc_real lambda1, mcmc_real T0,
                         mcmc_real psi, mcmc_real nu) {
    mcmc_real tanth = tan(theta0);
    mcmc_real y1;

    y1 = ak/((mcmc_real)16.0*myu*myu) * lambda1*lambda1
         * pow(T0, (mcmc_real)(-1.35)) * pow(nu, (mcmc_real)(-2.1));
    y1 *= (c_one + tanth*tanth) / (tanth*tanth);
    y1 *= (sin(c_two*theta0) * cos(c_two*psi)
           + c_two*theta0) / (c_two*theta0*theta0);
    y1 = pow(y1, c_1third);
    return y1;
}


__device__ mcmc_real Sv_func(mcmc_real theta0, mcmc_real lambda1, mcmc_real T0,
                             mcmc_real psi, mcmc_real r0, mcmc_real ymax,
                             mcmc_real nu, mcmc_real d) {
    mcmc_real Sv, y0, y1, thetam, Br;
    mcmc_real d2 = d*d;
    mcmc_real y00, y11, y111;

    y0 = r0*cos(psi);
    y1 = y1_func(theta0, lambda1, T0, psi, nu);
    thetam = atan2(tan(theta0), cos(psi));

    y00 = y0*y0;
    y11 = y1*y1;
    y111 = y11*y1;

    Br = planck(T0, nu);

    if (y1 < y0)
        Sv = (c_two*thetam*Br)/d2*(y111*(c_one/y0 - c_one/ymax));
    else if (y0 <= y1 && y1 < ymax)
        Sv = c_two*(thetam*Br)/d2*(y11 - y00)/c_two
             + y111*(c_one/y1 - c_one/ymax);
    else // if (y1 >= ymax)
        Sv = (thetam*Br)/d2*(ymax*ymax - y00);

    return Sv/(mcmc_real)1e-23; // to Jy
}
