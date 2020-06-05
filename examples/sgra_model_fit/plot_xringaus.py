#
# Plot brightness image of the 9-parameter model xrinfaus.
#
# Example:
#
# %run plot_xringaus.py   1.  25.  0.7  0.5   0.   1.   2.   0.3   0.
#
# Te parameters:         Zsp  Re   rq   ecc  fade gax   aq    gq   th
#                       (Jy) (uas)                                (rad)
#

from pylab import *
import matplotlib as mpl
from mpl_toolkits.mplot3d import axes3d
import os
import sys

import obsdatafit
reload(obsdatafit)
from obsdatafit import *
import mcmcfit1
reload(mcmcfit1)
import mcmcfit1 as mc

#
# This allows using LaTeX in the plot annotations
#
mpl.rcParams['mathtext.fontset'] = 'stix' # Allow math symbols b/w $ and $
mpl.rcParams['text.usetex'] = True # False     # True



UVspan = 160. # Glam. Set U and V span using brute force
ngrid = 100
nptotal = 9

Zsp, Re, rq, ecc, fade, gax, aq, gq, th = float_(sys.argv[1:])

vfparam = array([Zsp, Re, rq, ecc, fade, gax, aq, gq, th])

for i in xrange(nptotal): print '%g\t' % vfparam[i],
print
#print 'chi^2[0] = ', chi2[0]

Vi, Br, Ugrid, Vgrid, Xgrid, Ygrid, UVext, XYext = \
         model_vis2bri(xringaus, vfparam, UVspan, ngrid, ngrid, ngrid)
absVi = abs(Vi)
absBr = abs(Br)
phVi = arctan2(Vi.imag, Vi.real)
uext = UVext[:2]
vext = UVext[2:]



figure(figsize=(6,6));
imshow(absBr*1e3, origin='lower', cmap=cm.hot, extent=XYext, \
       interpolation='nearest'); grid(1)
title('Histogram Mean Model Brightness', fontsize=18)
colorbar(shrink=0.7)
xlabel(r'RA ($\mu$as)')
ylabel(r'Dec ($\mu$as)')
grid(1)
figtext(0.78, 0.795, 'mJy/pixel')
figtext(0.03, 0.94, r'$Z_{sp} = %4.2f$, $R_e = %4.1f$, $R_i/R_e = %4.2f$, $Ecc = %4.2f$' % (Zsp, Re, rq, ecc), fontsize=16)
figtext(0.03, 0.88, r'$Fade =  %4.2f$, $g_{ax} = %4.2f$, $a_q = %4.2f$, $g_q = %4.2f$, $\theta = %6.1f$' % (fade, gax, aq, gq, th), fontsize=16)

show()
