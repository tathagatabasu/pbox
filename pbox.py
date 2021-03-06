import numpy as np
import operator as op
import matplotlib.pyplot as plt
import scipy.stats as st

class cdf:
	"""Cumulative distribution function"""
	
	def __init__(self, xs):
		if np.size(xs) < 2:
			raise ValueError('length of cdf must be at least obe')
			
		self.xs = sorted(xs)
	
	def __len__(self):
		return len(self.xs)
	
	def min(self):
		return self.xs[0]
	
	def max(self):
		return self.xs[-1::]
	
	def print(self):
		print(self.xs)
	
	def mean(self):
		return np.mean(self.xs)
	
	def eval(self, x):
		n = len(self)
		for i in range(n):
			if x <= self.xs[i]:
				return i/n
			elif x > self.xs[(n-1)]:
				return 1
			
class pbox:
	"""pbox for modeling."""
	
	def __init__(self, u, l):
		if (np.size(u)!=np.size(l)):
			raise ValueError('lower bound and upper bound must have same length')
		elif(np.array_equal(np.minimum(u, l), u)==0):
			raise ValueError('lower bounds cannot be bigger than upper bounds')
		self.u = cdf(u)
		self.l = cdf(l)
		
	def __len__(self):
		return len(self.u)
	
	def min(self):
		return self.u.xs[0]
	
	def max(self):
		return self.l.xs[-1::]
	
	def print(self):
		print([self.u.xs , self.l.xs])
	
	def mean(self):
		return [cdf.mean(self.u), cdf.mean(self.l)]
	
	def eval(self,x):
		return [cdf.eval(self.u,x), cdf.eval(self.l,x)]
	
	def central_moment(self, k):
		
		m = self.mean()
		xs1 = self.l.xs - m[0]
		xs2 = self.u.xs - m[0]
		xs3 = self.l.xs - m[1]
		xs4 = self.u.xs - m[1]
		mat = np.matrix([xs1, xs2, xs3, xs4])
		mml = np.multiply(mat.min(axis=0), k)
		mmu = np.multiply(mat.max(axis=0), k)
		return [np.mean(np.minimum(mml,mmu)), np.mean(np.maximum(mml,mmu))]
	
	def sd(self):
		return self.central_moment(2)
	
	# unary negation
	def __neg__(self):
		uxs = np.negative(self.l.xs[::-1])
		lxs = np.negative(self.u.xs[::-1])
		return pbox(uxs, lxs)
	
	# logicals
	def __le__(self, e2):
		return self.l.xs[(len(self)-1)] <= e2
	def __lt__(self, e2):
		return self.l.xs[(len(self)-1)] < e2
	def __ge__(self, e2):
		return self.u.xs[0] >= e2
	def __gt__(self, e2):
		return self.u.xs[0] > e2
	
	# primitive operators
	def __add__(self, obj2):
		if (type(obj2) == int) or (type(obj2) == float):
			uxs = np.add(self.u.xs, obj2)
			lxs = np.add(self.l.xs, obj2)
			return pbox(uxs, lxs)
		else:
			raise ValueError('this operation is not defined')
		
	def __radd__(self, obj1):
		if (type(obj1) == int) or (type(obj1) == float):
			return self + obj1
		else:
			raise ValueError('this operation is not defined')
	
	def __sub__(self, obj2):
		return self + (-obj2)
	
	def __rsub__(self, obj1):
		return obj1 + (-self)
	
	def __mul__(self, obj2):
		if (self < 0) or (obj2 < 0):
			raise ValueError('inputs can not be negative')
		else:	
			if (type(obj2) == int) or (type(obj2) == float):
				uxs = np.multiply(self.u.xs, obj2)
				lxs = np.multiply(self.l.xs, obj2)
				return pbox(uxs, lxs)
			else:
				raise ValueError('this operation is not defined')
	
	def __rmul__(self, obj1):
		if (type(obj1) == int) or (type(obj1) == float):
			return self * obj1
		else:
			raise ValueError('this operation is not defined')
	
	def __rtruediv__(self, obj1):
		if (obj1 <= 0) or (self <= 0):
			raise ValueError('inputs must be strictly positive')
		else:
			uxs = np.divide(obj1, self.l.xs)[::-1]
			lxs = np.divide(obj1, self.u.xs)[::-1]
			return pbox(uxs, lxs)
		
	def __truediv__(self, obj2):
		if (self <= 0) or (obj2 <= 0):
			raise ValueError('inputs must be strictly positive')
		else:
			return self * (1/obj2)
	
	def __pow__(self, obj1):
		uxs = np.power(self.u.xs, obj1)
		lxs = np.power(self.l.xs, obj1)
		return pbox(uxs, lxs)
	
	def __rpow__(self, obj1):
		uxs = np.power(obj1, self.u.xs)
		lxs = np.power(obj1, self.l.xs)
		return pbox(uxs, lxs)
	
	def log(self):
		return pbox(np.log(self.u.xs), np.log(self.u.xs))
	
	def exp(self):
		return pbox(np.exp(self.u.xs), np.exp(self.l.xs))
	
	
	
	
	# implementation of Williamson & Downs convolution algorithm

	def indepconv(obj1, obj2, func):
		n = len(obj1)
		if n != len(obj2):
			raise ValueError('length of both pboxes must be same')
		uxs = pbox.sortfunc(obj1.u.xs, obj2.u.xs, func)
		lxs = pbox.sortfunc(obj1.l.xs, obj2.l.xs, func)
		iu = slice(0, n**2, n)
		il = slice((n-1), n**2, n)
		return pbox(uxs[iu], lxs[il])
	
	
	# operations with independent assumption
	
	def iadd(obj1, obj2):
		if pbox.ispbox(obj1) and pbox.ispbox(obj2):
			return pbox.indepconv(obj1, obj2, op.add)
		else:
			raise ValueError('inputs must be pbox')
	
	def isub(obj1, obj2):
		if pbox.ispbox(obj1) and pbox.ispbox(obj2):
			return pbox.iadd(obj1, -obj2)
		else:
			raise ValueError('inputs must be pbox')
	
	def imul(obj1, obj2):
		if pbox.ispbox(obj1) and pbox.ispbox(obj2):
			if (obj1 <= 0) or (obj2 <= 0):
				raise ValueError('inputs must be strictly positive')
			else:
				return pbox.indepconv(obj1, obj2, op.mul)
		else:
			raise ValueError('inputs must be pbox')
	
	def idiv(obj1, obj2):
		if pbox.ispbox(obj1) and pbox.ispbox(obj2):
			return pbox.imul(obj1, (1/obj2))
		else:
			raise ValueError('inputs must be pbox')
	
	# Frechet dependency
		
	def frechetconv(obj1, obj2, func):
		n = len(obj1)
		if n != len(obj2):
			raise ValueError('both pbox must have same length')
		uxs = [None] * n
		lxs = [None] * n
		for i in range(n):
			uxs[i] = max(func(obj1.u.xs[:i+1:],obj2.u.xs[:i+1:][::-1]))
			lxs[i] = min(func(obj1.l.xs[i::],obj2.l.xs[i::][::-1]))
		
		return pbox(uxs, lxs)
	
	# operations with Frechet dependency assumption
	
	def fadd(obj1, obj2):
		if pbox.ispbox(obj1) and pbox.ispbox(obj2):
			return pbox.frechetconv(obj1, obj2, np.add)
		else:
			raise ValueError('inputs must be pbox')
	
	def fmul(obj1, obj2):
		if pbox.ispbox(obj1) and pbox.ispbox(obj2):
			if (obj1 <= 0) or (obj2 <= 0):
				raise ValueError('inputs must be strictly positive')
			else:
				return pbox.frechetconv(obj1, obj2, np.multiply)
		else:
			raise ValueError('inputs must be pbox')
		
	def fsub(obj1, obj2):
		if pbox.ispbox(obj1) and pbox.ispbox(obj2):
			return pbox.fadd(obj1, -obj2)
		else:
			raise ValueError('inputs must be pbox')
	
	def fdiv(obj1, obj2):
		if pbox.ispbox(obj1) and pbox.ispbox(obj2):
			return pbox.fmul(obj1, 1/obj2)
		else:
			raise ValueError('inputs must be pbox')
	
	# Perfect convolution
	
	def perfectconv(obj1, obj2, func):
		n = len(obj1)
		if n != len(obj2):
			raise ValueError('both pbox must have same length')
		uxs = func(obj1.u.xs, obj2.u.xs)
		lxs = func(obj1.l.xs, obj2.l.xs)
		return pbox(uxs, lxs)
	
	# operations with perfect convolution
	
	def padd(obj1, obj2):
		if pbox.ispbox(obj1) and pbox.ispbox(obj2):
			return pbox.perfectconv(obj1, obj2, np.add)
		else:
			raise ValueError('inputs must be pbox')
	
	def psub(obj1, obj2):
		if pbox.ispbox(obj1) and pbox.ispbox(obj2):
			return pbox.padd(obj1, -obj2)
		else:
			raise ValueError('inputs must be pbox')
	
	def pmul(obj1, obj2):
		if pbox.ispbox(obj1) and pbox.ispbox(obj2):
			if (obj1 <= 0) or (obj2 <= 0):
				raise ValueError('inputs must be strictly positive')
			else:
				return pbox.perfectconv(obj1, obj2, np.multiply)
		else:
			raise ValueError('inputs must be pbox')
		
	def pdiv(obj1, obj2):
		if pbox.ispbox(obj1) and pbox.ispbox(obj2):
			return pbox.pmul(obj1, 1/obj2)
		else:
			raise ValueError('inputs must be pbox')
	
	# Opposite dependency
	
	def oppositeconv(obj1, obj2, func):
		n = len(obj1)
		if n != len(obj2):
			raise ValueError('both pbox must have same length')
		uxs = func(obj1.u.xs, obj2.u.xs[::-1])
		lxs = func(obj1.l.xs, obj2.l.xs[::-1])
		return pbox(uxs, lxs)
	
	# operations with opposite dependency
	
	def oadd(obj1, obj2):
		if pbox.ispbox(obj1) and pbox.ispbox(obj2):
			return pbox.oppositeconv(obj1, obj2, np.add)
		else:
			raise ValueError('inputs must be pbox')
	
	def osub(obj1, obj2):
		if pbox.ispbox(obj1) and pbox.ispbox(obj2):
			return pbox.oadd(obj1, -obj2)
		else:
			raise ValueError('inputs must be pbox')
	
	def omul(obj1, obj2):
		if pbox.ispbox(obj1) and pbox.ispbox(obj2):
			if (obj1 <= 0) or (obj2 <= 0):
				raise ValueError('inputs must be strictly positive')
			else:
				return pbox.oppositeconv(obj1, obj2, np.multiply)
		else:
			raise ValueError('inputs must be pbox')
		
	def odiv(obj1, obj2):
		if pbox.ispbox(obj1) and pbox.ispbox(obj2):
			return pbox.omul(obj1, 1/obj2)
		else:
			raise ValueError('inputs must be pbox')
	
	# p-box plot
	
	def plot(self, figno=0, xlim=[None, None], title='P-Box', block=False):
		xs_l = min(self.u.xs) - 10**(-5)
		xs_u = max(self.l.xs) + 10**(-5)
		xs = np.linspace(xs_l, xs_u, 200)
		fxs = [None]*200
		for i in np.linspace(0,199,200):
			fxs[int(i)]=pbox.eval(self, xs[int(i)])
		xmin = xlim[0]
		xmax = xlim[1]
		if xlim[0]==None:
			if xlim[1]==None:
				xmin = xs_l - 10**-1
				xmax = xs_u + 10**-1
			else:
				xmin = xs_l - 10**-1
		elif xlim[1]==None:
			xmax = xs_u + 10**-1
		plt.figure(figno)
		plt.plot(xs, fxs, 'k')
		plt.xlabel('x')
		plt.ylabel('CDF')
		plt.title(title)
		plt.xlim(xmin, xmax)
		plt.show(block)
		
	
	@staticmethod
	
	def ispbox(obj):
		return(type(obj)==pbox)
		
	def sortfunc(xs1, xs2, func):
		pass
		n1 = np.size(xs1)
		n2 = np.size(xs2)
		ys1 = np.tile(xs1, n2)
		ys2 = np.repeat(xs2, n1)
		ys = func(ys1, ys2)
		return sorted(ys)
	
	def norm(mean, sd, n=200):
		ps = np.linspace(0, 1, (n+1))
		ps[0] = min(0.001, 1/n)
		ps[n] = max(0.999, (1-1/n))
		if np.size(mean) == 1:
			mean = [mean]*2
		if np.size(sd) == 1:
			sd = [sd]*2
		xs = np.zeros(shape = (4,(n+1)))
		for j in range(4):
			for i in range(n+1):
				xs[j][i] = st.norm.ppf(ps[i],mean[j//2],sd[j%2])
		xs = np.matrix(xs)
		uxs = np.array(xs.min(0))[0].tolist()
		lxs = np.array(xs.max(0))[0].tolist()
		return pbox(uxs[:n:], lxs[1::])
	
	def uniform(low, up, n=200):
		ps = np.linspace(0, 1, (n+1))
		ps[0] = min(0.001, 1/n)
		ps[n] = max(0.999, (1-1/n))
		if np.size(low) == 1:
			low = [low]*2
		if np.size(up) == 1:
			up = [up]*2
		xs = np.zeros(shape = (4,(n+1)))
		for j in range(4):
			for i in range(n+1):
				xs[j][i] = st.uniform.ppf(ps[i],low[j//2],(up[j%2]-low[j//2]))
		xs = np.matrix(xs)
		uxs = np.array(xs.min(0))[0].tolist()
		lxs = np.array(xs.max(0))[0].tolist()
		return pbox(uxs[:n:], lxs[1::])
	
	def beta(a, b, n=200):
		ps = np.linspace(0, 1, (n+1))
		ps[0] = min(0.001, 1/n)
		ps[n] = max(0.999, (1-1/n))
		if np.size(a) == 1:
			a = [a]*2
		if np.size(b) == 1:
			b = [b]*2
		xs = np.zeros(shape = (4,(n+1)))
		for j in range(4):
			for i in range(n+1):
				xs[j][i] = st.beta.ppf(ps[i],a[j//2],b[j%2])
		xs = np.matrix(xs)
		uxs = np.array(xs.min(0))[0].tolist()
		lxs = np.array(xs.max(0))[0].tolist()
		return pbox(uxs[:n:], lxs[1::])
	
	def cauchy(n=200):
		ps = np.linspace(0, 1, (n+1))
		ps[0] = min(0.001, 1/n)
		ps[n] = max(0.999, (1-1/n))
		xs = [None]*(n+1)
		for i in range(n+1):
			xs[i] = st.cauchy.ppf(ps[i])
		return pbox(xs[:n:], xs[1::])
	
	def exponential(l, n=200):
		ps = np.linspace(0, 1, (n+1))
		ps[0] = min(0.001, 1/n)
		ps[n] = max(0.999, (1-1/n))
		if np.size(l) == 1:
			l = [l]*2
		xs = np.zeros(shape = (2,(n+1)))
		for j in range(2):
			for i in range(n+1):
				xs[j][i] = st.expon.ppf(ps[i], scale=(1/l[j//2]))
		xs = np.matrix(xs)
		uxs = np.array(xs.min(0))[0].tolist()
		lxs = np.array(xs.max(0))[0].tolist()
		return pbox(uxs[:n:], lxs[1::])
	
	def gamma(a, b, n=200):
		ps = np.linspace(0, 1, (n+1))
		ps[0] = min(0.001, 1/n)
		ps[n] = max(0.999, (1-1/n))
		if np.size(a) == 1:
			a = [a]*2
		if np.size(b) == 1:
			b = [b]*2
		xs = np.zeros(shape = (4,(n+1)))
		for j in range(4):
			for i in range(n+1):
				xs[j][i] = st.gamma.ppf(ps[i],a[j//2],scale=b[j%2])
		xs = np.matrix(xs)
		uxs = np.array(xs.min(0))[0].tolist()
		lxs = np.array(xs.max(0))[0].tolist()
		return pbox(uxs[:n:], lxs[1::])
	
	def invgamma(a, b, n=200):
		ps = np.linspace(0, 1, (n+1))
		ps[0] = min(0.001, 1/n)
		ps[n] = max(0.999, (1-1/n))
		if np.size(a) == 1:
			a = [a]*2
		if np.size(b) == 1:
			b = [b]*2
		xs = np.zeros(shape = (4,(n+1)))
		for j in range(4):
			for i in range(n+1):
				xs[j][i] = st.invgamma.ppf(ps[i],a[j//2],scale=b[j%2])
		xs = np.matrix(xs)
		uxs = np.array(xs.min(0))[0].tolist()
		lxs = np.array(xs.max(0))[0].tolist()
		return pbox(uxs[:n:], lxs[1::])
	
	def F(dfn, dfd, n=200):
		ps = np.linspace(0, 1, (n+1))
		ps[0] = min(0.001, 1/n)
		ps[n] = max(0.999, (1-1/n))
		if np.size(dfn) == 1:
			dfn = [dfn]*2
		if np.size(dfd) == 1:
			dfd = [dfd]*2
		xs = np.zeros(shape = (4,(n+1)))
		for j in range(4):
			for i in range(n+1):
				xs[j][i] = st.f.ppf(ps[i],dfn[j//2],dfd[j%2])
		xs = np.matrix(xs)
		uxs = np.array(xs.min(0))[0].tolist()
		lxs = np.array(xs.max(0))[0].tolist()
		return pbox(uxs[:n:], lxs[1::])
	
	def chi2(df, n=200):
		ps = np.linspace(0, 1, (n+1))
		ps[0] = min(0.001, 1/n)
		ps[n] = max(0.999, (1-1/n))
		if np.size(df) == 1:
			df = [df]*2
		xs = np.zeros(shape = (2,(n+1)))
		for j in range(2):
			for i in range(n+1):
				xs[j][i] = st.chi2.ppf(ps[i], df[j//2])
		xs = np.matrix(xs)
		uxs = np.array(xs.min(0))[0].tolist()
		lxs = np.array(xs.max(0))[0].tolist()
		return pbox(uxs[:n:], lxs[1::])
	
	def laplace(a, b, n=200):
		ps = np.linspace(0, 1, (n+1))
		ps[0] = min(0.001, 1/n)
		ps[n] = max(0.999, (1-1/n))
		if np.size(a) == 1:
			a = [a]*2
		if np.size(b) == 1:
			b = [b]*2
		xs = np.zeros(shape = (4,(n+1)))
		for j in range(4):
			for i in range(n+1):
				xs[j][i] = st.laplace.ppf(ps[i],a[j//2],b[j%2])
		xs = np.matrix(xs)
		uxs = np.array(xs.min(0))[0].tolist()
		lxs = np.array(xs.max(0))[0].tolist()
		return pbox(uxs[:n:], lxs[1::])
	
	def t(df, n=200):
		ps = np.linspace(0, 1, (n+1))
		ps[0] = min(0.001, 1/n)
		ps[n] = max(0.999, (1-1/n))
		if np.size(df) == 1:
			df = [df]*2
		xs = np.zeros(shape = (2,(n+1)))
		for j in range(2):
			for i in range(n+1):
				xs[j][i] = st.t.ppf(ps[i], df[j//2])
		xs = np.matrix(xs)
		uxs = np.array(xs.min(0))[0].tolist()
		lxs = np.array(xs.max(0))[0].tolist()
		return pbox(uxs[:n:], lxs[1::])

	def triangular(low, mode, up, n=200):
		ps = np.linspace(0, 1, (n+1))
		ps[0] = min(0.001, 1/n)
		ps[n] = max(0.999, (1-1/n))
		if np.size(low) == 1:
			low = [low]*2
		if np.size(up) == 1:
			up = [up]*2
		if np.size(mode) == 1:
			mode = [mode]*2
		xs = np.zeros(shape = (8,(n+1)))
		for j in range(8):
			for i in range(n+1):
				xs[j][i] = st.triang.ppf(ps[i],(mode[(j-(j//4)*4)//2]-low[j//4])\
										 /(up[j%2]-low[j//4]),low[j//4],(up[j%2]-low[j//4]))
		xs = np.matrix(xs)
		uxs = np.array(xs.min(0))[0].tolist()
		lxs = np.array(xs.max(0))[0].tolist()
		return pbox(uxs[:n:], lxs[1::])
	
