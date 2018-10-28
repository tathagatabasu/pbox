import numpy as np
import operator as op
import matplotlib.pyplot as plt
import scipy.stats as st
import pbox as p

# cdf checks

xs = np.linspace(0, 1, 11)

cdf1 = p.cdf(xs)
len(cdf1)
cdf1.mean()
cdf1.eval(0.5)

# pbox checks

uxs = np.linspace(0.1, 1.1, 11)
lxs = np.linspace(0.2, 1.2, 11)

pb1 = p.pbox(uxs, lxs)
len(pb1)
pb1.mean()
pb1.sd()
pb1.eval(.5)

pb2 = -pb1

# logicals

pb1 < 1
pb2 > 0

# scalar addition

pb3 = pb2 + 3

# independent assumption

pb1 + pb3
pb1 * pb3
pb1 - pb3
pb1 / pb3

# frechet assumptions
p.pbox.fadd(pb1, pb3)
p.pbox.fmul(pb1, pb3)
p.pbox.fsub(pb1, pb3)
p.pbox.fdiv(pb1, pb3)

# plot

p.pbox.plot(pb1)

# example checks

pb4 = p.pbox.uniform(0, 1, 40)
pb5 = pb4
pb6 = pb4 + pb5
# Williamson & Downs figure 21
p.pbox.plot(pb6)

pb7 = p.pbox.uniform(1, 2, 40)
pb8 = pb7
pb9 = p.pbox.fadd(pb7, pb8)
# Williamson & Downs figure 19
p.pbox.plot(pb9)