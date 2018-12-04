import numpy as np
import operator as op
import matplotlib.pyplot as plt
import scipy.stats as st
import pbox as p

# cdf checks

xs = np.linspace(0, 1, 11)

cdf1 = p.cdf(xs)
len(cdf1)
cdf1.min()
cdf1.max()
cdf1.print()
cdf1.mean()
cdf1.eval(0.5)

# pbox checks

uxs = np.linspace(0.1, 1.1, 11)
lxs = np.linspace(0.2, 1.2, 11)

pb1 = p.pbox(uxs, lxs)
len(pb1)
pb1.min()
pb1.max()
pb1.print()
pb1.mean()
pb1.sd()
pb1.eval(.5)

pb2 = -pb1

# logicals

pb1 < 1
pb2 > 0

# inbuilts

pb3 = pb2 + 3
p.pbox.plot(pb1 ** 2,1)

p.pbox.plot(2 ** pb1,2)

p.pbox.plot(p.pbox.log(pb1),3)
p.pbox.plot(p.pbox.exp(pb1),4)

# independent assumption
p.pbox.iadd(pb1, pb3)
p.pbox.imul(pb1, pb3)
p.pbox.isub(pb1, pb3)
p.pbox.idiv(pb1, pb3)

# frechet assumptions
p.pbox.fadd(pb1, pb3)
p.pbox.fmul(pb1, pb3)
p.pbox.fsub(pb1, pb3)
p.pbox.fdiv(pb1, pb3)

# perfect dependency
p.pbox.plot(p.pbox.padd(pb1, pb3),5)
p.pbox.plot(p.pbox.psub(pb1, pb3),6)
p.pbox.plot(p.pbox.pmul(pb1, pb3),7)
p.pbox.plot(p.pbox.pdiv(pb1, pb3),8)

# opposite dependency
p.pbox.plot(p.pbox.oadd(pb1, pb3),9)
p.pbox.plot(p.pbox.osub(pb1, pb3),10)
p.pbox.plot(p.pbox.omul(pb1, pb3),11)
p.pbox.plot(p.pbox.odiv(pb1, pb3),12)

# plot

p.pbox.plot(pb1)

# distributions

p.pbox.plot(p.pbox.norm([0,1], [1,2]),13)
p.pbox.plot(p.pbox.uniform([0,2], [3,6]),14)
p.pbox.plot(p.pbox.triangular([0,1], [1,2], [2,3]),15, block=True)
p.pbox.plot(p.pbox.beta([1,2], [1, 3]),16)
p.pbox.plot(p.pbox.cauchy(),17)
p.pbox.plot(p.pbox.exponential([1, 2]),18)
p.pbox.plot(p.pbox.gamma([1,2], [1, 3]),19)
p.pbox.plot(p.pbox.invgamma([1,2], [1, 3]),20)
p.pbox.plot(p.pbox.F([10,12], [12,14]),21)
p.pbox.plot(p.pbox.chi2([5,6]),22)
p.pbox.plot(p.pbox.laplace([1,2], [2,3]),23)
p.pbox.plot(p.pbox.t(10),24)


# example checks

pb4 = p.pbox.uniform(0, 1, 40)
pb5 = pb4
pb6 = p.pbox.iadd(pb4, pb5)
# Williamson & Downs figure 21
p.pbox.plot(pb6,25)

pb7 = p.pbox.uniform(1, 2, 40)
pb8 = pb7
pb9 = p.pbox.fadd(pb7, pb8)
# Williamson & Downs figure 19
p.pbox.plot(pb9,26)

# Ferson & Siegrist figure 4

pbu1 = p.pbox.uniform(1, 25)
pbu2 = p.pbox.fadd(pbu1, pbu1)
p.pbox.plot(pbu2,27, [0, 50], 'frechet dependency')
pbu2 = p.pbox.padd(pbu1, pbu1)
p.pbox.plot(pbu2,28, [0, 50], 'perfect dependency')
pbu2 = p.pbox.oadd(pbu1, pbu1)
p.pbox.plot(pbu2,29, [0, 50], 'opposite dependency')
pbu2 = p.pbox.iadd(pbu1, pbu1)
p.pbox.plot(pbu2,30, [0, 50], 'independence')

