#
# search_Rosenbrock_api.py
#
# Using MCMC-RE on GPU finds the global minimum of the Rosenbrock function
# (aka "Rosenbrock valley" or "Rosenbrock banana function").
#
# The global minimum, z = 0, is at (1,1) inside a long, narrow, parabolic
# shaped flat valley. To find the valley is trivial. To converge to the
# global minimum, however, is a difficult optimization task.
#
# The function is defined by
#    f(x,y)=(1 - x)^2 + 100(y - x^2)^2
#
# This is the low-level API example: it calls mcmc_interf.mcmcuda() directly.
#

import mcmc_interf as mi
import numpy as np
import time
import sys
from numpy import pi, array, float32, int32, uint8, int64, uint64, where, \
     zeros, ones, arange, ones_like, linspace, sqrt, argsort, meshgrid, \
	 argsort, where, logspace
from matplotlib.pyplot import figure, plot, subplot, show, hist, grid, axis, \
	xlabel, ylabel, title, cm
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# mpl.rcParams['text.usetex'] = True
# mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #if needed

#
# MCMC_REAL is float64 or float32 depending on how mcmc_interf was built.
# It must match the .pyx (mcmc_interf_float.pyx / mcmc_interf_double.pyx)
# and the C/CUDA side (mcmc_real in mcmc_types.h).
#
MCMC_REAL = np.float64 if mi.MCMC_DOUBLE_FLAG else np.float32

nseq = int32(14)   # Number of processing cores (CUDA blocks)
nbeta = int32(32)  # Number of temperatures
niter = int32(1000)
nburn = int32(400) # Number of initial iters ignored due to the transience
nstates = int32(nseq*nbeta)
nseqit = int32(nseq*niter)
seed = uint64(np.trunc(1e6*time.time()%(10*nstates)))
imodel = 0

pdescr = array((1,  1),  dtype=int32)
ptotal = array((0., 0.), dtype=MCMC_REAL)
nptot = len(ptotal)
pmint = array((-5., -5.), dtype=MCMC_REAL)
pmaxt = array((5., 5.),   dtype=MCMC_REAL)


# ivar[nprm] maps prm[0:nprm] on ptotal[0:nptot] (not needed here )
ivar = where(pdescr != 0)[0] # Maps prm[0:nprm] on ptotal[0:nptot]
ivar = ivar.astype(int32)
nprm = len(ivar)
pmin = pmint[ivar]
pmax = pmaxt[ivar]
# invar[nptot] maps ptotal[0:nptot] on prm[0:nprm] (not needed here )
invar = zeros(nptot, dtype=int32)
ip = 0
for i in range(nptot):
    if pdescr[i] != 0: 
        invar[i] = ip
        ip = ip + 1
    else:
        invar[i] = -1

nadj = int32(100)
beta1 = MCMC_REAL(1.)
betan = MCMC_REAL(0.0001)   # ~exp(chi2/2)

npass = 1

coor =  array((2, 2, 0.25, 0.5), dtype=MCMC_REAL); ncoor = len(coor)
icoor = zeros(1,                 dtype=int32);   nicoor = len(coor)
dat =   zeros(1, dtype=MCMC_REAL); ndat = len(dat) # Rosenbrock min value = 0
std2r = ones_like(dat,           dtype=MCMC_REAL)
idat =  zeros(1,                 dtype=int32);   nidat = len(idat)
datm =  zeros((nbeta*nseq*ndat), dtype=MCMC_REAL); ndatm = len(datm)
chi2m = zeros((nbeta,nseq,ndat), dtype=MCMC_REAL)
terms = zeros((nbeta,nseq,ndat), dtype=MCMC_REAL)
rndst = zeros((nbeta,nseq,48),   dtype=uint8)

beta = zeros((nbeta), dtype=MCMC_REAL)
pstp = zeros((nprm,nbeta,nseq), dtype=MCMC_REAL)  # Adaptive step size
pout = zeros((nprm,nbeta,nseq,niter), dtype=MCMC_REAL)
tout = zeros((nbeta,nseq,niter), dtype=int32)
chi2 = zeros((nseq,niter), dtype=MCMC_REAL)
rate_acpt = zeros((nprm,nbeta), dtype=MCMC_REAL)
rate_exch = zeros((nbeta), dtype=MCMC_REAL)
flag = zeros((nbeta,nseq), dtype=int32)  
tcur = zeros((nbeta,nseq), dtype=int32)  
chi2c = zeros((nbeta,nseq), dtype=MCMC_REAL)  # old chi^2 array
n_cnt =  zeros((nprm,nbeta,nseq), dtype=int32) # number of Metropolis trials
n_acpt = zeros((nprm,nbeta,nseq), dtype=int32)
n_exch = zeros((nbeta,nseq), dtype=int32)
n_hist = zeros((nadj,nprm,nbeta,nseq), dtype=int32) # accept/reject history
ptentn = zeros((nbeta,nseq), dtype=int32)   # Tentative parameter numbers
ptent =  zeros((nbeta,nseq), dtype=MCMC_REAL) # Tentative parameters
pcur = zeros((nbeta,nseq,nptot), dtype=MCMC_REAL) # Accepted param. set

# Fill in the working storage for threads calculating model visibilities
for i in range(nbeta):
    for j in range(nseq):
        pcur[i,j,:] = ptotal
        for k in range(nprm):
            pcur[i,j,ivar[k]] = pmin[k] + (pmax[k] - pmin[k])/MCMC_REAL(2.)

# Initialize parameter steps
pmid = (pmax - pmin)/MCMC_REAL(50.)
for i in range(nprm):
    pstp[i,:,:] = pmid[i]

print('pstp[:,0,0] = ', pstp[:,0,0]) 

# Initialize tcur
n0_nbeta = arange(nbeta, dtype=int32) # 0..nbeta-1
for iseq in range(nseq):
    tcur[:,iseq] = n0_nbeta

# Initialize temperatures
if nbeta == 1:
    bstp = MCMC_REAL(1.0)  # For debugging only
else:
    bstp = (betan/beta1)**(MCMC_REAL(1.0)/MCMC_REAL(nbeta-1))
for i in range(nbeta):
    beta[i] = beta1*bstp**i

# Initialize counter
n_cnt[:] = np.int32(1)

print('std2r = ', std2r, ', idat = ', idat, ', nidat = ', nidat)
#sys.exit(0)

mi.mcmcuda(coor, dat, std2r, icoor, idat, datm, chi2m, \
           pdescr, ivar, invar, ptotal, pmint, pmaxt, \
           pout, tout, chi2, chi2c, tcur, flag, \
           n_acpt, n_exch, n_cnt, n_hist, beta, \
           pstp, ptent, ptentn, pcur, rndst, \
           uint64(seed), int32(imodel), \
           int32(ncoor), int32(ndat), int32(nicoor), int32(nidat), \
		   int32(ndatm), int32(nptot), int32(nprm), \
           int32(nadj), int32(npass), int32(nbeta), \
		   int32(nseq), int32(nburn), int32(niter))

mi.reset_gpu()


c2 = chi2.flatten()
po = pout[:,0,:,:].reshape((nprm,nseq*niter))
im = c2.argmin()   # Indices of the chi^2 minimums 

print('The minimum chi^2 at chi2[%d] = %f' % (im, c2[im]))
print('pout[%d] = ' % (im), po[:,im])

#
# Print the best root pairs found
#
imin = where(c2 < 0.00001)[0]
chi2_min = c2[imin]
pout_best = po[:,imin]
imsrt = argsort(chi2_min)
chi2_min = chi2_min[imsrt]
pout_best = pout_best[:,imsrt]
n_best = len(chi2_min)

print()
print('Exact solution: (x,y) = (%10f,%10f)' % (1, 1))
print()
print('Best minimum found:')
print('     x           y       chi^2')
for im in range(n_best):
	print('%10f %10f %10f' % (pout_best[0,im], pout_best[1,im], chi2_min[im]))

	
figure(figsize=(12,6))
subplot(121); hist(po[0,:], 50, color='b'); grid(1)
xlabel(r'$x$', fontsize=20)
title('MCMC Output Distribution for X')

subplot(122); hist(po[1,:], 50, color='b'); grid(1)
xlabel(r'$y$', fontsize=20)
title('MCMC Output Distribution for Y')

#
# Plot contours and 3D surface of the Rosenbrock function
#

ros = lambda x, y: (1 - x)**2 + 100.*(y - x**2)**2

x = linspace(-3., 3., 1000)
y = linspace(-1.5, 10., 1000)
X, Y = meshgrid(x, y)

z = ros(X, Y)

v = logspace(-0.5, 3.5, 10)

fig = plt.figure(figsize=(14,6))
ax2 = fig.add_subplot(121)
#ax2.contour(X, Y, z, 500, cmap=cm.coolwarm)
cs = ax2.contour(X, Y, z, v, cmap=cm.coolwarm)
#ax2.clabel(cs, v[5:], colors='k', fmt='%3.0f')
ax2.plot(1, 1, 'ro')
ax2.set_xlabel(r'$x$', fontsize=20)
ax2.set_ylabel(r'$y$', fontsize=20)
ax2.set_title(r'', fontsize=20)
grid(1);

x = linspace(-3., 3., 40)
y = linspace(-1.5, 10., 40)
X, Y = meshgrid(x, y)

z = ros(X, Y)

ax3 = fig.add_subplot(122, projection='3d')
ax3.plot_surface(X, Y, z, cmap=cm.coolwarm, rstride=1, cstride=1, alpha=0.8)
ax3.plot(x, x**2, (1-x)**2, color='b', lw=2)
ax3.plot([1], [1], [0], markerfacecolor='r', marker='o') #, markersize=10)
ax3.set_xlabel(r'$x$', fontsize=20)
ax3.set_ylabel(r'$y$', fontsize=20)
ax3.set_zlabel(r'$z$', fontsize=20)
fig.text(.43, 0.95, r'Rosenbrock Valley', fontsize=20, family='serif')



show()