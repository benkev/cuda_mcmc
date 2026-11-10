/*
 * mcmc_types.h
 *
 * Единые типы для cuda_mcmc.
 *
 * Переключение float/double — макросом MCMC_DOUBLE при компиляции:
 *   make MCMC_DOUBLE=1     -> mcmc_real = double
 *   make                   -> mcmc_real = float
 *
 * Критичные для точности массивы (dat, datm, chi2m, chi2c, chi2,
 * pcur, pout, ptent, pstp, beta, std2r, coor, ptotal, pmint, pmaxt)
 * имеют тип mcmc_real.
 *
 * Служебные массивы (rndst, индексы, флаги, счётчики) — фиксированные
 * типы mcmc_rng / mcmc_int, независимо от MCMC_DOUBLE.
 */

#ifndef MCMC_TYPES_H
#define MCMC_TYPES_H

#ifdef MCMC_DOUBLE
  typedef double mcmc_real;
  #define MCMC_REAL_STR "double"
#else
  typedef float  mcmc_real;
  #define MCMC_REAL_STR "float"
#endif

/* Служебные типы — всегда фиксированы */
typedef float          mcmc_aux;   /* для служебных float, если понадобится */
typedef unsigned char  mcmc_rng;   /* состояние curandState — 48 байт */
typedef int            mcmc_int;   /* индексы, счётчики, флаги */

#endif /* MCMC_TYPES_H */ 
