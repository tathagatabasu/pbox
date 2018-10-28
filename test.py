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
p.pbox.plot(pb1 ** 2)

p.pbox.plot(2 ** pb1)

p.pbox.plot(p.pbox.log(pb1))
p.pbox.plot(p.pbox.exp(pb1))

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
p.pbox.plot(p.pbox.padd(pb1, pb3))
p.pbox.plot(p.pbox.psub(pb1, pb3))
p.pbox.plot(p.pbox.pmul(pb1, pb3))
p.pbox.plot(p.pbox.pdiv(pb1, pb3))

# opposite dependency
p.pbox.plot(p.pbox.oadd(pb1, pb3))
p.pbox.plot(p.pbox.osub(pb1, pb3))
p.pbox.plot(p.pbox.omul(pb1, pb3))
p.pbox.plot(p.pbox.odiv(pb1, pb3))

# plot

p.pbox.plot(pb1)

# distributions

p.pbox.plot(p.pbox.norm(0, 1))
p.pbox.plot(p.pbox.uniform(0, 1))
p.pbox.plot(p.pbox.triangular(0, 1, 2))
p.pbox.plot(p.pbox.beta(1, 1))
p.pbox.plot(p.pbox.cauchy())
p.pbox.plot(p.pbox.exponential(2))
p.pbox.plot(p.pbox.gamma(1, 1))
p.pbox.plot(p.pbox.invgamma(1, 1))
p.pbox.plot(p.pbox.F(12, 14))
p.pbox.plot(p.pbox.chi(5))
p.pbox.plot(p.pbox.laplace(1, 2))
p.pbox.plot(p.pbox.t(10))


# example checks

pb4 = p.pbox.uniform(0, 1, 40)
pb5 = pb4
pb6 = p.pbox.iadd(pb4, pb5)
# Williamson & Downs figure 21
p.pbox.plot(pb6)

pb7 = p.pbox.uniform(1, 2, 40)
pb8 = pb7
pb9 = p.pbox.fadd(pb7, pb8)
# Williamson & Downs figure 19
p.pbox.plot(pb9)

