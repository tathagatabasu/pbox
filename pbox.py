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
	
	def mean(self):
		return np.mean(self.xs)
	
	def eval(self, x):
		n = len(self)
		for i in range(n):
			if x < self.xs[i]:
				return i/n
			elif x >= self.xs[(n-1)]:
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
	
	# implementation of Williamson & Downs convolution algorithm

	def conv(obj1, obj2, func):
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
		return pbox.conv(obj1, obj2, op.add)
	
	def isub(obj1, obj2):
		return pbox.iadd(obj1, -obj2)
	
	def imul(obj1, obj2):
		return pbox.conv(obj1, obj2, op.mul)
	
	def idiv(obj1, obj2):
		return pbox.imul(obj1, (1/obj2))
	
	# Frechet dependency
		
	def frechet(obj1, obj2, func):
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
		return pbox.frechet(obj1, obj2, np.add)
	
	def fmul(obj1, obj2):
		if (obj1 <= 0) or (obj2 <= 0):
			raise ValueError('inputs must be strictly positive')
		else:
			return pbox.frechet(obj1, obj2, np.multiply)
		
	def fsub(obj1, obj2):
		return pbox.fadd(obj1, -obj2)
	
	def fdiv(obj1, obj2):
		return pbox.fmul(obj1, 1/obj2)
	
	# p-box plot
	
	def plot(self):
		xs_l = min(self.u.xs) - 10**(-5)
		xs_u = max(self.l.xs) + 10**(-5)
		xs = np.linspace(xs_l, xs_u, 200)
		fxs = [None]*200
		for i in np.linspace(0,199,200):
			fxs[int(i)]=pbox.eval(self, xs[int(i)])
		
		plt.plot(xs, fxs, 'k')
		plt.xlabel('x')
		plt.ylabel('CDF')
		plt.show()
		
	
	@staticmethod
		
	def sortfunc(xs1, xs2, func):
		pass
		n1 = np.size(xs1)
		n2 = np.size(xs2)
		ys1 = np.tile(xs1, n2)
		ys2 = np.repeat(xs2, n1)
		ys = func(ys1, ys2)
		return sorted(ys)
	
	def norm(mean, sd, n):
		ps = np.linspace(0, 1, (n+1))
		ps[0] = min(0.001, 1/n)
		ps[n] = max(0.999, (1-1/n))
		xs = [None]*(n+1)
		for i in range(n+1):
			xs[i] = st.norm.ppf(ps[i],mean,sd)
		return pbox(xs[:n:], xs[1::])
	
	def uniform(low, up, n):
		ps = np.linspace(0, 1, (n+1))
		ps[0] = min(0.001, 1/n)
		ps[n] = max(0.999, (1-1/n))
		xs = [None]*(n+1)
		for i in range(n+1):
			xs[i] = st.uniform.ppf(ps[i],low,(up-low))
		return pbox(xs[:n:], xs[1::])
	
	def triangular(low, mode, up, n):
		ps = np.linspace(0, 1, (n+1))
		ps[0] = min(0.001, 1/n)
		ps[n] = max(0.999, (1-1/n))
		xs = [None]*(n+1)
		for i in range(n+1):
			xs[i] = st.triang.ppf(ps[i],(mode-low)/(up-low),low,(up-low))
		return pbox(xs[:n:], xs[1::])
	