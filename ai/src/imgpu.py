###########################################################
#                                                         #
# imgpu.py                                                #
#                                                         #
# Module for fitting image models to radiointerferometer  #
# data. The Markov Chain Monte-Carlo (MCMC) method is     #
# used. The MCMC is implemented as a set of "kernels"     #
# written in CUDA C language.                             #
#                                                         #
# Created: 14 May 2013 by Leonid Bevkevitch               #
###########################################################
#
import numpy as np
import os, sys, re
import mcmc_interf as mi
import time
from time import strptime
from calendar import timegm
import traceback

PK2I = 2**15   # 2 integers in 2x15=30-bit integer (a1, a2)
PK3I = 2**10   # 3 integers in 3x10=30-bit integer (i1, i2, i3)

#
# MCMC_REAL is float or double, depending on how mcmc_interf was built.
# It is used for all critical arrays: dat, datm, chi2m, chi2c, chi2,
# pcur, pout, ptent, pstp, beta, std2r, coor, ptotal, pmint, pmaxt.
# Service arrays (rndst, indices, counters, flags) always use np.uint8/np.int32.
#
MCMC_REAL = np.float64 if mi.MCMC_DOUBLE_FLAG else np.float32
MCMC_REAL_STR = "float64" if mi.MCMC_DOUBLE_FLAG else "float32"

#
# On the NVIDIA GPUs the fastest computations are made using 32-bit integers
# and 32-bit floats (in C/C++ 'int' and 'float' types).
# However, on modern 64-bit PSs the Python implementation both basic types,
# 'int' and 'float' are 64-bit by default. So, to avoid repeated writing
# (and, what is much worse, forgetting to write) np.int32(number), some
# frequently used small integers are assigned the aliases like below:
#
short1 = np.int32(1)
short2 = np.int32(2)
short3 = np.int32(3)
short4 = np.int32(4)
short6 = np.int32(6)
short10 = np.int32(10)


class Mcgpu(object):   # The new-style class must inherit the 'object' class
    """
    The class for universal use of MCMC-RE on GPU.
    Provides initializations for some arrays to reduce amount of the user code.
    The methods provided are the same as of its descendant, Mcgpu_Sgra (the
    latter overloads them).

    Input parameters not present in mcmc_interf.mcmcuda():

    beta1,
    betan: parameters of the Python classes imgpu and imgpu_sgra.
           They set the values in the array of temperatures beta[nbeta]
           from beta[0] = beta1 to beta[nbeta-1] = betan.
           The temperatures correspond to $\\beta = 1/{kT}$.
           The lowest temperature, beta[0], is usually set to beta1 = 1.
           The highest, beta[nbeta-1], can be set to several orders of
           magnitude less, something like betan = 0.0001
           The values in beta[nbeta] fall off exponentially.
    """
    def __init__(self, coor=None, dat=None, std2r=None, icoor=None, idat=None, \
                 pdescr=None, ptotal=None, pmint=None, pmaxt=None, \
                 beta=None, beta1=None, betan=None, \
                 seed=None, imodel=0, npass=1, \
                 nadj=100, nbeta=32, nseq=14,  nburn=300, niter=500):

        # MCMC_REAL: critical arrays use MCMC_REAL (float32 or float64)
        if dat is None:
            self.dat = np.zeros(1, dtype=MCMC_REAL)
        else:
            self.dat = dat.astype(MCMC_REAL)
        ndat = np.int32(np.size(self.dat))

        #
        # If the reciprocals of data variances are not given, they are
        # assumed unities
        #
        if std2r is None:
            self.std2r = np.ones_like(self.dat, dtype=MCMC_REAL)
        else:
            self.std2r = std2r.astype(MCMC_REAL)

        if coor is None:
            self.coor = np.zeros(1, dtype=MCMC_REAL)
        else:
            self.coor = coor.astype(MCMC_REAL)
        ncoor = np.int32(np.size(self.coor))

        if icoor is None:
            self.icoor = np.zeros(1, dtype=np.int32)
        else:
            self.icoor = icoor.astype(np.int32)
        nicoor = np.int32(np.size(self.icoor))

        if idat is None:
            self.idat = np.zeros(1, dtype=np.int32)
        else:
            self.idat = idat.astype(np.int32)
        nidat = np.int32(np.size(self.idat))

        # MCMC_REAL
        self.datm = np.zeros((nbeta*nseq*ndat), dtype=MCMC_REAL)
        ndatm = np.int32(np.size(self.datm))

        # MCMC_REAL
        self.chi2m = np.zeros((nbeta,nseq,ndat), dtype=MCMC_REAL)

        self.pdescr = pdescr.astype(np.int32)
        self.nptot = nptot = np.int32(len(pdescr))

        # MCMC_REAL
        if ptotal is None:
            self.ptotal = np.zeros(nptot, dtype=MCMC_REAL)
        else:
            self.ptotal = ptotal.astype(MCMC_REAL)

        # MCMC_REAL
        self.pmint = pmint.astype(MCMC_REAL)
        self.pmaxt = pmaxt.astype(MCMC_REAL)

        #
        # Find the number of parameters nprm as number of pdescr != 0.
        # ivar[nprm] maps prm[0:nprm-1] on ptotal[0:nptot-1]
        # ivar[] contains indices into ptotal of the mutable parameters
        #
        self.ivar = np.where(pdescr != 0)[0]
        self.ivar = ivar = self.ivar.astype(np.int32)
        self.nprm = nprm = np.int32(len(self.ivar))
        self.pmin = pmin = self.pmint[self.ivar]
        self.pmax = pmax = self.pmaxt[self.ivar]

        #
        # invar[nptot] maps ptotal[0:nptot-1] on prm[0:nprm-1].
        # invar[], at the locations corresponding to those in ptotal,
        # contains indices into prm[nprm], an array of the mutable
        # parameters. At the locations where pdescr is zero, invar has "-1".
        #
        self.invar = np.zeros(nptot, dtype=np.int32)
        ip = 0
        for i in range(self.nptot):
            if self.pdescr[i] != 0:
                self.invar[i] = ip
                ip = ip + 1
            else:
                self.invar[i] = -1

        #
        # If the random number generator seed is not specified,
        # set it to a random, time dependent value.
        #
        if seed is None:
            nstates = np.int32(nseq*nbeta)
            self.seed = np.uint64(np.trunc(1e6*time.time()%(10*nstates)))
        else:
            self.seed = np.uint64(seed)

        self.ncoor = np.int32(ncoor)
        self.ndat = np.int32(ndat)
        self.nidat = np.int32(nidat)
        self.nicoor = np.int32(nicoor)
        self.ndatm = np.int32(ndatm)
        self.nadj = np.int32(nadj)
        self.nbeta = np.int32(nbeta)
        self.npass = np.int32(npass)
        self.nseq = np.int32(nseq)
        self.nburn = np.int32(nburn)
        self.niter = np.int32(niter)
        self.imodel = np.int32(imodel)

        # MCMC_REAL: all these arrays participate in chi^2 / alpha calculations
        self.pstp = np.zeros((nprm,nbeta,nseq), dtype=MCMC_REAL)
        self.pout_4d = np.zeros((nprm,nbeta,nseq,niter), dtype=MCMC_REAL)
        self.tout = np.zeros((nbeta,nseq,niter), dtype=np.int32)
        self.chi2_2d = np.zeros((nseq,niter), dtype=MCMC_REAL)
        self.flag = np.zeros((nbeta,nseq), dtype=np.int32)
        self.tcur = np.zeros((nbeta,nseq), dtype=np.int32)
        self.chi2c = np.zeros((nbeta,nseq), dtype=MCMC_REAL)
        self.n_cnt =  np.zeros((nprm,nbeta,nseq), dtype=np.int32)
        self.n_acpt = np.zeros((nprm,nbeta,nseq), dtype=np.int32)
        self.n_exch = np.zeros((nbeta,nseq), dtype=np.int32)
        self.n_hist = np.zeros((nadj,nprm,nbeta,nseq), dtype=np.int32)
        self.ptentn = np.zeros((nbeta,nseq), dtype=np.int32)
        self.ptent =  np.zeros((nbeta,nseq), dtype=MCMC_REAL)
        self.pcur = np.zeros((nbeta,nseq,nptot), dtype=MCMC_REAL)
        self.rndst = np.zeros((nbeta,nseq,48),   dtype=np.uint8)

        # Set initial parameter values to averages along the prior dimensions
        for i in range(nbeta):
            for j in range(nseq):
                self.pcur[i,j,:] = self.ptotal
                for k in range(nprm):
                    self.pcur[i,j,ivar[k]] = self.pmin[k] + \
                             (self.pmax[k] - self.pmin[k])/MCMC_REAL(2.)

        # Initialize parameter steps
        pmid = (self.pmax - self.pmin)/MCMC_REAL(50.)
        for i in range(nprm):
            self.pstp[i,:,:] = pmid[i]

        # Initialize tcur, temperature indices
        n0_nbeta = np.arange(nbeta, dtype=np.int32) # 0..nbeta-1
        for iseq in range(nseq):
            self.tcur[:,iseq] = n0_nbeta

        # Initialize temperatures
        if beta1 is None:
            self.beta1 = MCMC_REAL(1.)
        else:
            self.beta1 = MCMC_REAL(beta1)

        if betan is None:
            self.betan = MCMC_REAL(0.0001)
        else:
            self.betan = MCMC_REAL(betan)

        if beta is None:
            self.beta = np.zeros((nbeta), dtype=MCMC_REAL)
            beta1 = MCMC_REAL(1.)
            betan = self.betan   # ~exp(chi2/2)
            bstp = (betan/beta1)**(MCMC_REAL(1.0)/MCMC_REAL(nbeta-1))
            for i in range(nbeta):
                self.beta[i] = beta1*bstp**i
        else:
            self.beta = beta.astype(MCMC_REAL)

        # Initialize counter
        self.n_cnt[:] = np.int32(1)


    def burnin_and_search(self):

        mi.mcmcuda(self.coor, self.dat, self.std2r, \
            self.icoor, self.idat, self.datm, self.chi2m, \
            self.pdescr, self.ivar, self.invar, self.ptotal,
            self.pmint, self.pmaxt, \
            self.pout_4d, self.tout, self.chi2_2d, self.chi2c, \
            self.tcur, self.flag, \
            self.n_acpt, self.n_exch, self.n_cnt, self.n_hist, self.beta, \
            self.pstp, self.ptent, self.ptentn, self.pcur, \
            self.rndst, self.seed, self.imodel, \
            self.ncoor, self.ndat, self.nicoor, self.nidat, self.ndatm, \
            self.nptot, self.nprm, self.nadj, self.npass, self.nbeta, \
            self.nseq,  self.nburn, self.niter)

        mi.reset_gpu()

        #
        # Rates of acceptance and of exchange
        #
        nitsq = MCMC_REAL(self.niter*self.nseq)
        self.rate_acpt = sum(self.n_acpt, 2)/nitsq
        self.rate_exch = sum(self.n_exch, 2)/nitsq

        #
        # Select only the coldest (beta==0) temperature from self.pout_4d[] and:
        # 1. Turn self.pout_4d[nprm,0,nseq,niter] into a two-dimensional
        #    self.pout[nprm,nseq*niter];
        # 2. Turn self.chi2_2d[nseq,niter] into a one-dimensional
        #    self.chi2[nseq*niter].
        # After that, each chi^2 value at a self.chi2[] location will
        # correspond to the parameter set location in the second dimension
        # of self.pout[].
        #
        self.chi2 = self.chi2_2d.flatten()
        self.pout = self.pout_4d[:,0,:,:]
        self.pout = self.pout.reshape((self.nprm,self.nseq*self.niter))

        return 0


    def calcmodchi2(self, prm_in, imodel=None):
        """
        Parallel computation of the  __global__ calc_chi2_terms()
        function from the file gpu_mcmc.cu for the array of given
        parameters prm_in. It may be
        1D, prm_in[nptot], array-like variable: 1D (one set of nptot
            parameters),
        2D, prm_in[n,nptot] (a linear array of parameter vectors), or
        3D, prm_in[nbeta,nsec,nptot].
        The sizes nbeta and nsec are retained in the returned arrays
        datm and chi2m

        The results returned:
        datm, chi2m
        """
        if type(prm_in) is not np.ndarray:     # If prm is a list or a tuple
            prm = np.array(prm_in, dtype=MCMC_REAL)
        else:
            prm = prm_in.astype(MCMC_REAL)

        sh = prm.shape
        if   prm.ndim == 1:
            nbeta = 1; nseq = 1; nptot = sh[0]
        elif prm.ndim == 2:
            nbeta = sh[0]; nseq = 1; nptot = sh[1]
        elif prm.ndim == 3:
            nbeta = sh[0]; nseq = sh[1]; nptot = sh[2]
        else:
            print("calcmodchi2(): input parameter array has > 3 or < 1 " \
                  "dimensions")
            return

        ndat = self.ndat
        ndatm = nbeta*nseq*ndat
        pcur = prm.reshape((nbeta,nseq,nptot))
        # MCMC_REAL
        datm =  np.zeros((ndatm), dtype=MCMC_REAL)
        chi2m = np.zeros((nbeta,nseq,ndat), dtype=MCMC_REAL)
        chi2c = np.zeros((nbeta,nseq), dtype=MCMC_REAL)
        flag = np.ones((nbeta,nseq), dtype=np.int32)

        imod = self.imodel if imodel == None else imodel

        mi.calcmodchi2(self.coor, self.dat, self.std2r,
                       self.icoor, self.idat, \
                       datm, chi2m, pcur, chi2c, flag, imod, \
                       self.ncoor, self.ndat, self.nicoor, self.nidat, \
                       ndatm, nptot, nbeta, nseq)

        mi.reset_gpu()

        #
        # Remove degenerate, single-dimensional entries
        # from the shape of the output arrays
        #
        if len(sh) < 3:
            datm = datm.squeeze()
            chi2m = chi2m.squeeze()

        return datm, chi2m


    def reset_gpu(self):
        mi.reset_gpu()

    def set_gpu(self, devNum=0):
        er = mi.set_gpu(devNum)
        if er != 0:
            print("GPU not set to %d. cudaSetDevice(devNum) error: %d" %
                  (devNum, er))


#==========================================================================
#                        END OF CLASS Mcgpu(object)
#==========================================================================


class Mcgpu_Sgra(Mcgpu):
    """
    The class representing the model fitting procedures and data.
    """
    def __init__(self, uvfile=None, imodel=1, npass=2,
                 nseq=14, nbeta=32, nburn=100, niter=100,
                 seed=1234, beta1=1., betan = 0.0001,
                 pdescr = None, ptotal=None, pmint=None, pmaxt=None,
                 ulam=None, vlam=None, blin=None,
                 amp=None, phase=None, cphs=None, cpix=None, uvcp=None,
                 eamp=None, ecph=None,
                 tsec=None, chan=None, wei=None, wlam=None,
                 nadj=100, use_cphs=True):

        super(Mcgpu_Sgra,self).__init__(
                 pdescr=pdescr, ptotal=ptotal, pmint=pmint, pmaxt=pmaxt, \
                 beta1=beta1, betan=betan, seed=seed, imodel=imodel, \
                 npass=npass, nadj=nadj, nbeta=nbeta, nseq=nseq, \
                 nburn=nburn, niter=niter)

        self.error = ''
        self.uvfile = uvfile
        self.imodel = imodel
        # MCMC_REAL for floating arrays
        self.ulam = None if ulam == None else np.asarray(ulam, dtype=MCMC_REAL)
        self.vlam = None if vlam == None else np.asarray(vlam, dtype=MCMC_REAL)
        self.wlam = None if wlam == None else np.asarray(wlam, dtype=MCMC_REAL)
        self.uvcp = None if uvcp == None else np.asarray(uvcp, dtype=MCMC_REAL)
        self.amp = None if amp == None else np.asarray(amp, dtype=MCMC_REAL)
        self.phase = None if phase == None else np.asarray(phase, dtype=MCMC_REAL)
        self.eamp = None if eamp == None else np.asarray(eamp, dtype=MCMC_REAL)
        self.ecph = None if ecph == None else np.asarray(ecph, dtype=MCMC_REAL)
        self.blin = None if blin == None else np.asarray(blin, dtype=np.int32)
        self.tsec = None if tsec == None else np.asarray(tsec, dtype=MCMC_REAL)
        self.fsig = None
        self.chan = chan
        self.wei = None if wei == None else np.asarray(wei, dtype=MCMC_REAL)

        self.use_cphs = bool(use_cphs)  # If True, closure phases are used.

        if uvfile != None:
            self.ulam, self.vlam, self.wlam, self.amp, self.phase, \
                self.blin, self.tsec, self.fsig, self.chan, self.wei = \
                   read_uvdata(uvfile)
            self.ulam = MCMC_REAL(1e-3)*self.ulam
            self.vlam = MCMC_REAL(1e-3)*self.vlam
        else:
            print("Warning: uvfile with the UV coverage data not specified")

        if all(self.fsig) == 0:  self.fsig[:] = MCMC_REAL(1.)
        self.eamp = np.ones_like(self.amp, dtype=MCMC_REAL)
        if eamp is None:
            if all(self.fsig) != 0:
                self.eamp = np.copy(self.fsig)
            else:
                self.error = "Some sigmas in uv file are zeros."
                print("Error: "+self.error)
                return
        elif np.isscalar(eamp):
            if eamp != 0:
                self.eamp[:] = MCMC_REAL(eamp)
            else:
                self.error = "1-sigma visibility amplitude error eamp is zero."
                print("Error: "+self.error)
                return
        elif len(eamp) == len(self.amp):
            if all(eamp) != 0:
                self.eamp = np.copy(np.asarray(eamp, dtype=MCMC_REAL))
            else:
                self.error = "Some of eamp elements are zero."
                print("Error: "+self.error)
                return
        else:
            self.error = "number of 1-sigma errors in parameter eamp (%d) " \
                  "differs from number of visibility amplitudes " \
                  "in self.amp (%d)." % (len(eamp), len(self.amp))
            print("Error: "+self.error)
            return

        self.base = np.sqrt(self.ulam**2 + self.vlam**2)  # Baseline lengths
        self.cphs = None if cphs == None else np.asarray(cphs, dtype=MCMC_REAL)
        self.cpix = None if cpix == None else np.asarray(cpix, dtype=np.int32)
        self.trid = None
        self.tcphsec = None

        zeroPhase = all(self.phase == 0)
        if zeroPhase and cphs == None:
            self.use_cphs = False  # If no phases, no closures used

        self.nvis = nvis = np.int32(len(self.amp))
        self.nmph = nmph = nbeta*nseq*nvis

        self.cpExt = 0

        if self.use_cphs:  # Closure phases used
            if cphs == None:
                self.cpExt = 0
                self.cphs, self.cpix, self.trid, self.tcphsec = \
                           calc_closures(self.phase, self.tsec, self.blin)
            else:
                self.cpExt = 1

            self.ncph = ncph = np.int32(len(self.cphs))
            self.vicp = np.hstack((self.amp, self.cphs))
            self.nvicp = nvicp = nvis + ncph
            self.nchi2m = nchi2m = nbeta*nseq*nvicp
            self.npass = np.int32(2)
        else: # No closure phases used
            self.ncph = ncph = 0
            self.vicp = np.copy(self.amp)
            self.nvicp = nvicp = nvis
            self.nchi2m = nchi2m = nbeta*nseq*nvis

        if self.use_cphs:  # Closure phases used
            self.ecph = np.ones(ncph, dtype=MCMC_REAL)
            if ecph is None:
                self.error = "Closure phase 1-sigma error(s) not provided " \
                      "in parameter ecph."
                print("Error: "+self.error)
                return
            elif np.isscalar(ecph):
                if ecph != 0:
                    self.ecph[:] = MCMC_REAL(ecph)
                else:
                    self.error = "Closure phase 1-sigma error ecph is zero."
                    print("Error: "+self.error)
                    return
            elif len(ecph) == len(self.ecph):
                if all(ecph) != 0:
                    self.ecph = np.copy(np.asarray(ecph, dtype=MCMC_REAL))
                else:
                    self.error = "Some of ecph elements are zero."
                    print("Error: "+self.error)
                    return
            else:
                self.error = "number of 1-sigma errors in parameter " \
                      "ecph (%d) differs from number of closure phaese " \
                      "in self.cphs (%d)." % (len(ecph), len(self.cphs))
                print("Error: "+self.error)
                return
        else:
            self.ecph = None

        #
        # ====================== MCMC Algorithm Data =====================
        #
        nptot = self.nptot
        nprm = self.nprm

        self.ndat = ndat = nvis + ncph
        self.nidat = nidat = short3
        self.ndatm = ndatm = nbeta*nseq*(ndat + nvis)
        self.seed = np.uint64(seed)

        # MCMC_REAL
        self.dat =   np.empty((ndat),  dtype=MCMC_REAL)
        self.std2r = np.ones((ndat), dtype=MCMC_REAL)
        self.std2r[:nvis] = MCMC_REAL(1.)/(self.eamp**2)
        if use_cphs:
            self.std2r[nvis:] = MCMC_REAL(1.)/(self.ecph**2)
            if cphs == None:
                self.ncoor = ncoor = short2*nvis
                self.nicoor = nicoor = short3*ncph
            else:
                self.ncoor = ncoor = short2*nvis + short4*ncph
                self.nicoor = nicoor = 0
        else:
            self.ncoor = ncoor = short2*nvis
            self.nicoor = nicoor = 0

        # MCMC_REAL
        self.coor =  np.empty((ncoor), dtype=MCMC_REAL)
        self.icoor = np.empty(nicoor, dtype=np.int32)
        self.idat =  np.empty(nidat, dtype=np.int32)
        self.datm =  np.zeros((ndatm), dtype=MCMC_REAL)
        self.chi2m = np.zeros((nbeta,nseq,ndat), dtype=MCMC_REAL)

        # MCMC_REAL
        self.mvicp = np.zeros((nbeta,nseq,ndat), dtype=MCMC_REAL)
        self.mpha = np.zeros((nbeta,nseq,nvis), dtype=MCMC_REAL)

        # Packing arrays
        self.coor[:(2*nvis)] = np.vstack((self.ulam, self.vlam)).T.flatten()

        if self.use_cphs:  # Closure phases used
            if cphs != None:
                self.coor[(2*nvis):] = self.uvcp.flatten()
            if not self.cpExt:
                self.icoor[:] = self.cpix.flatten()
            self.dat[:nvis] = self.amp
            self.dat[nvis:] = self.cphs
        else:
            self.dat[:] = np.copy(self.amp)

        self.idat[:] = nvis, nchi2m, self.cpExt


    def burnin_and_search(self):

        if self.error != "":
            print("Cannot burnin and search: first fix the error: "+self.error)
            return 1

        mi.mcmcuda(self.coor, self.dat, self.std2r, \
            self.icoor, self.idat, self.datm, self.chi2m, \
            self.pdescr, self.ivar, self.invar, self.ptotal,
            self.pmint, self.pmaxt, \
            self.pout_4d, self.tout, self.chi2_2d, self.chi2c, \
            self.tcur, self.flag, \
            self.n_acpt, self.n_exch, self.n_cnt, self.n_hist, self.beta, \
            self.pstp, self.ptent, self.ptentn, self.pcur, \
            self.rndst, self.seed, self.imodel, \
            self.ncoor, self.ndat, self.nicoor, self.nidat, self.ndatm, \
            self.nptot, self.nprm, self.nadj, self.npass, self.nbeta, \
            self.nseq,  self.nburn, self.niter)

        mi.reset_gpu()

        # Parts of self.datm used to calculate self.chi2m:
        self.mvicp = self.datm[:self.nchi2m]. \
                      reshape((self.nbeta,self.nseq,self.ndat)) # Head
        self.mpha = self.datm[self.nchi2m:]. \
                    reshape((self.nbeta,self.nseq,self.nvis))  # Tail

        # Rates of acceptance and of exchange
        nitsq = MCMC_REAL(self.niter*self.nseq)
        self.rate_acpt = sum(self.n_acpt, 2)/nitsq
        self.rate_exch = sum(self.n_exch, 2)/nitsq

        return 0


    def calcmodchi2(self, prm_in, imodel=None):
        """
        Calculate model visibilities and closure phases
        for the array of given parameters prm. It may be
        array-like variable: 1D (one set of 9 parameters),
        2D (a linear array of parameter vectors), or
        3D, prm[nbeta,nsec,nptot]

        The results:
        chi2c, mvicp, mpha, chi2m, datm
        """
        if self.error != "":
            print("First fix the error: "+self.error)
            return 1

        if type(prm_in) != np.ndarray:
            prm = np.array(prm_in, dtype=MCMC_REAL)
        else:
            prm = prm_in.astype(MCMC_REAL)

        sh = prm.shape
        if len(sh) == 1:     # 1D
            nbeta = 1; nseq = 1; nptot = sh[0]
        elif len(sh) == 2:   # 2D
            nbeta = sh[0]; nseq = 1; nptot = sh[1]
        elif len(sh) == 3:   # 3D
            nbeta = sh[0]; nseq = sh[1]; nptot = sh[2]
        else:
            print("calcmodchi2(): input parameter array has > 3 or < 1 " \
                  "dimensions")
            return

        ndat = self.ndat
        nvis = self.nvis
        pcur = prm.reshape((nbeta,nseq,nptot))
        ndatm = nbeta*nseq*(ndat + nvis)
        # MCMC_REAL
        datm =  np.zeros((ndatm), dtype=MCMC_REAL)
        chi2m = np.zeros((nbeta,nseq,ndat), dtype=MCMC_REAL)
        chi2c = np.zeros((nbeta,nseq), dtype=MCMC_REAL)
        flag = np.ones((nbeta,nseq), dtype=np.int32)

        if imodel == 1 or imodel == 2: imod = imodel
        else:                          imod = self.imodel

        mi.calcmodchi2(self.coor, self.dat, self.std2r,
                       self.icoor, self.idat, \
                       datm, chi2m, pcur, chi2c, flag, imod, \
                       self.ncoor, self.ndat, self.nicoor, self.nidat, \
                       ndatm, nptot, nbeta, nseq)

        mi.reset_gpu()

        nchi2m = nbeta*nseq*ndat

        print('nbeta, nseq, ndat, nchi2m = ', nbeta, nseq, ndat, nchi2m)

        mvicp = datm[:nchi2m]. \
                      reshape((nbeta,nseq,ndat))
        mpha = datm[nchi2m:]. \
                    reshape((nbeta,nseq,nvis))

        if len(sh) == 1:
            chi2c = chi2c[0,0]
            mvicp = mvicp.flatten()
            mpha = mpha.flatten()
            chi2m = chi2m.flatten()

        return chi2c, mvicp, mpha, chi2m, datm


    def calcmodel(self, ulam, vlam, prm_in, imodel=2):
        """
        Calculate model visibilities at (ulam,vlam) points
        for the array of vectors of given parameters prm_in.
        """
        if type(prm_in) != np.ndarray:
            prm = np.array(prm_in, dtype=MCMC_REAL)
        else:
            prm = prm_in.astype(MCMC_REAL)

        ul = ulam.flatten()
        vl = vlam.flatten()
        uv_sh = ulam.shape
        udim = uv_sh[0]
        vdim = uv_sh[1]

        prm_sh = prm.shape

        if len(prm_sh) == 1:     # 1D
            nbeta = 1; nseq = 1; nptot = prm_sh[0]
            mamp_sh = uv_sh
        elif len(prm_sh) == 2:   # 2D
            nbeta = prm_sh[0]; nseq = 1; nptot = prm_sh[1]
            mamp_sh = (nbeta,udim,vdim)
        elif len(prm_sh) == 3:   # 3D
            nbeta = prm_sh[0]; nseq = prm_sh[1]; nptot = prm_sh[2]
            mamp_sh = (nbeta,nseq,udim,vdim)
        else:
            print("calcmodel(): input parameter array has > 3 or < 1 " \
                  "dimensions")
            return

        if imodel == 1 or imodel == 2: imod = imodel
        else:                          imod = self.imodel

        pcur = prm.reshape((nbeta,nseq,nptot))

        ndat = nvis = len(ul)
        ncoor = short2*nvis
        ndatm = nbeta*nseq*short2*nvis
        nmamp = nbeta*nseq*nvis

        # MCMC_REAL
        datm =  np.zeros((ndatm), dtype=MCMC_REAL)
        coor =  np.empty((ncoor), dtype=MCMC_REAL)
        coor[:nvis] = ul; coor[nvis:] = vl;
        idat = np.array((nvis, 0, 0), dtype=np.int32)
        nidat = len(idat)

        mi.calcmodel(coor, idat, datm, pcur, imod,  \
                     ncoor, ndat, nidat, ndatm, nptot, nbeta, nseq)

        mi.reset_gpu()

        mamp = datm[:nmamp].reshape(mamp_sh)
        mpha = datm[nmamp:].reshape(mamp_sh)

        return mamp, mpha


#==========================================================================
#                        END OF CLASS Mcgpu_Sgra(Mcgpu)
#==========================================================================




def cbinom(n, m):
    """
    Returns value of m-th binomial coefficient of (x+y)^n expansion.
    Or, which is the same, the number of m-subsets of a set with
    n elements
    Note: the returned coefficient type is int.
    """
    if n < 0 or m < 0 or m > n:
        return -1
    if m == 0 or m == n:
        return 1
    if m == 1 or m == n-1:
        return n
    num = n
    den = m
    if m > 2:
        for i in range(m-1,1,-1):
            den = den*i
    if n > 2:
        for i in range(n-1,n-m,-1):
            num = num*i
    return num//den



def combgen(n, m):
    """
    Generates an array C[n,m] of all the subsets of size m
    out of the set of n elements. Returns the array C[csize,m]
    of subsets.
    The C[] lines are indices into a sequence s of arbitrary items,
    pointing at those of them that are chosen for a subset.
    However, the most convenient are sets specified as arrays,
    because arrays allow group indexing.
    For example, say, we have a set s = np.array((10, 14, 7, 3, 8, 25)).
    After the call
        c = combgen(6,3)
    each line of c has 3 indices into s. All the c lines enumerate
    all the s subsets. For instance:
        c[5,:] = np.array([0, 2, 4])
    then
        s[c[5,:]] is
           np.array([10,  7,  8]).
    
    Thus, the array of indices returned by combgen can be used for
    obtaining subsets of sets containing elements of virtually any type. 
    """

    #-------------------------------------- Local function
    
    def cgenrecurs(S, n, m, p, C, ic):
    ## """
    ## Generates the subsets of size m from the set S[n] starting
    ## from the p-th position. Returns the index into array C where
    ## the subsets are stored.
    ## This local recursive function is intended to be only used
    ## in the wrapping combgen() function.
    ## """
        if p < m:
            if p == 0:
                for i in range(n):
                    S[0] = i
                    ic = cgenrecurs(S, n, m, 1, C, ic)
            else: # Here p > 0:
                for i in range(S[p-1]+1,n):
                    S[p] = i
                    ic = cgenrecurs(S, n, m, p+1, C, ic)
        else: # Here p == m
            C[ic,:] = S[:m]
            ic = ic + 1
        return ic
    #---------------------------------------- End of local function

    csize = cbinom(n, m)
    S = np.zeros((n), int)         # Set of indices
    C = np.zeros((csize, m), int)  # Result: all the subsets' indices
    ic = 0                         # Current position in C for writing 
    ic = cgenrecurs(S, n, m, 0, C, ic) 
    return C



def find_triangles(antennas):
    """
    This function takes as its input the array of antennas, and returns
    an array filled with all possible unique triangles and a 1D array
    of the triangle IDs. Call:
    triangles, triids = find_triangles(antennas)
    """
    antarr = np.array(antennas)        # Convert to an array
    nant = np.size(antarr)
    itri = combgen(nant, 3)         # Indices of all possible antenna triplets
    ntri = np.size(itri,0)             # Number of triangles
    tri = np.empty(itri.shape, int)    # Array of triangles
    trid = np.empty(ntri, int)         # Array of unique IDs for the triangles 
    for i in range(ntri): 
        tri[i,:] = antarr[itri[i,:]]  # Triangle of antennae
        # A triangle ID is made as a radix maxantnum number whose
        # "digits" are the antenna numbers in the triangle.
        # It is unique for each triangle. The ID is used to speed up
        # searches for a specific triangle in the triangle array.
        trid[i] = tri[i,0] + PK3I*(tri[i,1] + PK3I*tri[i,2])
    return tri, trid



def get_time_indices(tsec):
    """
    Find all the equal-time time spans in the tsec array.
    Returns two arrays of the indices into the tsec array (row numbers),
    ibeg[] and iend[], each pointing at the beginning and the end of an
    equal-time data span.
    
    """
    #
    # Get pointers to starts of the equal-time intervals.
    # The last points past the end of tsec array - for consistency.
    #
    ntsec = len(tsec)
    itim = np.where(np.diff(tsec))[0] + 1 # 1 needed to fix location error
    itim = np.hstack((0, itim, ntsec)) # Starts at 0 at the beginning
    #
    # The itim[] now points at *beginnings* of the equal-time data pieces,
    # but we still do not know where the *last* piece (span?) ends.
    # The first difference of itim contains lengths of equal-time spans.
    # Find the last 1 in the itim 1st differences, ditim[]:
    #
    ditim =  np.diff(itim)     # Lengths of eq-time spans
    nditim = len(ditim)

    #print 'ntsec = ', ntsec, ', nitim = ', len(itim), ', nditim = ', nditim
    #print 'itim = \n', itim
    #print 'ditim = \n', ditim

    #
    # Gather all pointers to the intervals longer than 2 in the
    # lists ibeg and iend
    #
    i = 0
    ibeg = []
    iend = []
    while i < nditim:  # nditim = nitim - 1, so itim[i+1] is valid
        if ditim[i] >= 3:
            ibeg.append(itim[i])
            iend.append(itim[i+1])
        i = i + 1
        
    return ibeg, iend



def calc_closures(phase, tsec, blin):
    """
    Calculate closure phases from the data returned by readamp2():
    ul, vl, amp, phase, tsec, blin = readamp???(amp_file)
    blin[:,2] - pairs of antenna numbers starting drom 1.
    ## blid must be an int array of baseline indices, bl1 + 0x200000*bl2.
    Returns: 
    cphs: closure phases
    ## cpid: closure phase indices, dtype=int64
    ##       cpid is calculated as i1 + 0x200000*(i2 + i3 *0x200000),
    ##       where 0x200000 = 2097152 = 2**21,
    ##       and i1, i2, and i3 are indices into ul, vl, amp and phase.
    ##       So, a closure phase can be calculated as
    ##       cphs = phase[i1] + phase[i2] - phase[i3] for a given cpid.
    trid: triangle indices.
          trid is calculated as a1 + 0x200000*(a2 + a3*0x200000),
          where 0x200000 = 2097152 = 2**21,
          and a1, a2, and a3 are numbers of the stations in a triangle.
    tcphsec: closure phase times in seconds
    If phase has length 0 to 2, returns cphs, cpid, and tcphsec with zero length
    
    calc_closures() is a replacement for readcphs2():
    cphs, cuv, tri, tcphsec  = readcphs2(cphs_file)
    where tcphsec is 'tsec for closure phases'.
    """
    if len(phase) >= 3:
        # Get starts and ends of equal-time spans > 3 baselines long
        ibeg, iend = get_time_indices(tsec) 
        ntimes = len(ibeg)  # Number of same-time data segments > 3 bls
      
    if len(phase) < 3 or ntimes == 0: # No triangles
        cphs = np.array([], dtype=np.float32)
        #cpid = np.array([], dtype=np.int64)
        cpix = np.array([], dtype=np.int32)
        trid = np.array([], dtype=np.int32)
        tcphsec = np.array([], dtype=np.float32)
        return cphs, cpid, trid, tcphsec  #================================>>>

    times = tsec[ibeg]            # Unique times
    #bls = np.vstack((blid%PK2I, blid/PK2I)).T
    bls = blin
    antennas = np.unique(bls)
    triangles, triids = find_triangles(antennas)
    ntriangles = len(triangles[:,0])
    maxcps = ntimes*ntriangles   # Maximal number of closure phases
    cphs = np.zeros(maxcps, np.float32)
    cpix = np.zeros((maxcps,3), np.int32)
    trid = np.zeros((maxcps), np.int32)
    tcphsec = np.zeros((maxcps), float) # Closure phase times in seconds

    iclp = 0
    for itim in range(ntimes):    # Consider ith equal-time span
        i0 = ibeg[itim]           # Span start in data[]
        i1 = iend[itim]           # Span end   in data[]
        nbl = i1 - i0             # Number of data rows in the span
        bl = np.int_(bls[i0:i1,:])   # Convert i0:i1 baseline array to int
        ant = np.unique(bl)          # Leave only sorted unique antenna numbers
        nant = len(ant)           # Number of antenne employed in the i0-i1 span
        # Find all possible triangles in the span
        itri = combgen(nant, 3)   # Indices into ant to get triangles
        ntri = np.size(itri,0)       # Number of triangles in this span

        for j in range(ntri): # Over all triangles in this span
            a1, a2, a3 = ant[itri[j,:]] # Triangle of antennae
            # Find indices of the three baselines made of the antennae
            ib1 = i0 + np.where((bl[:,0] == a1) & (bl[:,1] == a2))[0][0]
            ib2 = i0 + np.where((bl[:,0] == a1) & (bl[:,1] == a3))[0][0]
            ib3 = i0 + np.where((bl[:,0] == a2) & (bl[:,1] == a3))[0][0]
            #print 'ib1, ib2, ib3 = ', ib1, ib2, ib3
            # Phases in 
            ph1 = phase[ib1]
            ph2 = phase[ib2]
            ph3 = phase[ib3]


            if True:   # ph1 != 0. and ph2 != 0. and ph3 != 0.:
                ph = ph1 + ph2 - ph3
                cphs[iclp] = np.arctan2(np.sin(ph), np.cos(ph))
                trid[iclp] = a1 + PK3I*(a2 + PK3I*a3)
                tcphsec[iclp] = times[itim]
                # Get unique closure phase ID
                # ib1, ib2, ib3 are indices into ul, vl, amp, and phase
                #cpid[iclp] = ib1 + PK3I*(ib2 + PK3I*ib3)
                cpix[iclp,:] = ib1, ib2, ib3
                iclp = iclp + 1

            # print "phase[%d]=%g,\t phase[%d]=%g,\t phase[%d]=%g,\t "  \
        #  "cph=%g,\t cphs[%d]=%g" % (ib1, ph1, ib2, ph2, ib3, ph3, ph, \
            #                             iclp-1, cphs[iclp-1])

    cphs = cphs[:iclp]
    #cpid = cpid[:iclp]
    cpix = cpix[:iclp,:]
    trid = trid[:iclp]
    tcphsec = tcphsec[:iclp]

    return cphs, cpix, trid, tcphsec        # End of calc_closures



def count_lines(filename, numcount=None):
    """
    Returns number of lines in text file. If run under unix/linux
    or any other POSIX-type system, it uses the fastest utility wc:
    wc -l lilename
    Otherwise Python I/O is used.
    The numcount is a number of numbers on the line in integer,
    floating point, or exponential format required for the line
    to be counted as valid. The numbers can be separated by
    arbitrary characters, say
    2009-10-17 21:15:56.21 3.62 03-08 is
    parsed as 9 numbers:
    2009, 10, 17, 21, 15, 56.21 3.62 03, 08
    """
    if os.name == 'posix' and numcount == None: # The fastest way:
        print(os.popen('wc -l ' + filename).read().split())
        lines = int(os.popen('wc -l ' + filename).read().split()[0])
        return lines
    # If not POSIX system, count using native python means:
    elif os.name != 'posix' and numcount == None:
        n = 1
        f = open(filename, 'r')
        lines = 0
        for line in f:
            lines += 1
        f.close()
        return lines
    elif numcount != None:
        # Regex pattern to match any numeric string, be it integer or float
        p = re.compile(r'[+-]?\d+\.?\d*[eE][+-]?\d+|'\
                       r'[+-]?\d*\.?\d+[eE][+-]?\d+|'\
                       r'[+-]?\d+\.?\d*|[+-]?\d*\.?\d+|[+-]?\d+');
        # Count only the lines with exactly numcount numbers
        f = open(filename, 'r')
        nrows = 0
        while True:
            row = f.readline()
            if row == '': break  #=============== Exit upon EOF ============>>>
            nums = p.findall(row)
            if len(nums) == numcount:
                nrows += 1
        f.close()
        return nrows
    return -1  # In what case?..                    # End of count_lines()



def read_uvdata(uv_file):
    """
    Read amplitude data file into float32 arrays:
    ulam, vlam, wlam, amp, phase, blin, tsec, sig, chan, wei
    where tsec is time in seconds since beginning of *Epoch*, and
    blin baselines as pairs of antenna numbers
    ### blid - baseline IDs
    ###    (to get antenna #s: ant1 = blid%PK2I, ant2 = blid/PK2I)

    Here is a snippet of the time, u, v, amp, phase file
    
    Scan Start (UT)            U(klam)      V(klam)      W(klam)  Baseline 
    Channel         Visibility (amp, phase)      Weight   Sigma
     2008:263:05:50:40.12   -1969263.57    496717.85   2920931.16  01-02     
       00        (     0.000000,     0.000000)     0.00    0.00000
     2008:263:05:50:40.12   -1529174.41      5210.19   2693549.21  01-03
       00        (     0.000000,     0.000000)     0.00    0.00000
     2008:263:05:50:40.12   -2687487.50   1809532.77   3124965.72  02-03
       00        (     0.000000,     0.000000)     0.00    0.00000
   
    Combines count_lines() and readamp().
    Returns ulam, vlam, wlam, amp, phase, blin, tsec, sig, chan, wei
    """
    # Regex pattern to match any numeric string, be it integer or float
    p = re.compile(r'[+-]?\d+\.?\d*[eE][+-]?\d+|'\
                   r'[+-]?\d*\.?\d+[eE][+-]?\d+|' \
                   r'[+-]?\d+\.?\d*|[+-]?\d*\.?\d+|[+-]?\d+')
    
    n = count_lines(uv_file)
    
    amp =   np.zeros(n, np.float32)
    phase = np.zeros(n, np.float32)
    sig =   np.zeros(n, np.float32)
    ulam =  np.zeros(n, np.float32)
    vlam =  np.zeros(n, np.float32)
    wlam =  np.zeros(n, np.float32)
    wei =   np.zeros(n, np.float32)
    tsec =  np.zeros(n, np.float32)
    chan =  np.zeros(n, np.int32)
    #blid =  np.zeros(n, np.int32)  # Important! WHY????????????????????
    blin =  np.zeros((n,2), np.int32)
    
    fp = open(uv_file, "r");

    i = 0
    while True:
        row = fp.readline()
        if row == '':
            break
        nums = p.findall(row)  # Fish out all the numbers from row into nums 
        if len(nums) != 15:
            continue #================ non-data line; do nothing =====>>>
      
        nums = np.array(nums)     # Array of strings; just to ease indexing
        u, v, w, chan[i], amp[i], phase[i], weig, sig[i] = \
           np.float_(nums[[5,6,7,10,11,12,13,14]])

        if weig != 0.0:
            wei[i] = weig
            ulam[i] = 1e-3*u    # From kilolambda to Megalambda
            vlam[i] = 1e-3*v    # From kilolambda to Megalambda
            wlam[i] = 1e-3*w    # From kilolambda to Megalambda
            phase[i] = np.radians(phase[i])
            # row[:20] is like 2009:095:02:40:00.00 or so
            tsec[i] = timegm(strptime(row[:17], '%Y:%j:%H:%M:%S')) # + \
            if row[18:20].isdigit():
                tsec[i] = tsec[i] + 0.01*np.float32(row[18:20])
            a1 =  int(nums[8])
            a2 = -int(nums[9])
            blin[i,:] = a1, a2
            # blid[i] = a1 + PK2I*a2     # Get unique baseline ID
        i = i + 1
        
    fp.close()
    
    if i == 0:
        #
        # Signal by returning empty arrays
        #
        ulam  = np.array([], dtype=np.int32)
        vlam  = np.array([], dtype=np.int32)
        wlam  = np.array([], dtype=np.int32)
        amp   = np.array([], dtype=np.int32)
        phase = np.array([], dtype=np.int32)
        chan  = np.array([], dtype=np.int32)
        tsec  = np.array([], dtype=np.int32)
        wei   = np.array([], dtype=np.int32)
        sig   = np.array([], dtype=np.int32)
        blin  = np.array([], dtype=np.int32)
        return ulam, vlam, wlam, amp, phase, chan, tsec, wei, sig, blin
      
    ulam.resize(i)
    vlam.resize(i)
    wlam.resize(i)
    amp.resize(i)
    phase.resize(i)
    sig.resize(i)
    tsec.resize(i)
    blin.resize((i,2))
   
    return ulam, vlam, wlam, amp, phase, blin, tsec, sig, chan, wei
                                                   # end of def read_uvdata()



