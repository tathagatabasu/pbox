import numpy as np
import operator as op

class cdf:
	"""CUmulative distribution function"""
	
	def __init__(self, xs):
		if np.size(xs) < 2:
			raise ValueError('length of cdf must be at least obe')
			
		self.xs = sorted(xs)
	
	def __len__(self):
		return len(self.xs)
	
	def mean(self):
		return np.mean(self.xs)
	
	def eval(self, x):
		n = self.len()
		for i in range(n):
			if x < self.xs[i]:
				return 1/n
			elif x >= self.xs[(n-1)]:
				return 1
			
class pbox:
	
	def __init__(self, u, l):
		if (np.size(u)!=np.size(l)):
			raise ValueError('lower bound and upper bound must have same length')
		elif(np.array_equal(np.minimum(u, l), u)==0):
			raise ValueError('lower bounds cannot be bigger than upper bounds')
		self.u = cdf(u)
		self.l = cdf(l)
		
	def __len__(self):
#		return cdf.len(self.u)
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
	
	def __neg__(self):
		uxs = np.negative(self.l.xs[::-1])
		lxs = np.negative(self.u.xs[::-1])
		return pbox(uxs, lxs)
	
	def __le__(self, e2):
		return self.l.xs[(self.len()-1)] <= e2
	def __lt__(self, e2):
		return self.l.xs[(self.len()-1)] < e2
	def __ge__(self, e2):
		return self.u.xs[0] >= e2
	def __gt__(self, e2):
		return self.u.xs[0] > e2
	
	# implementation of Williamson & Downs, Figure 14, page 127

	def sortfunc(self, xs1, xs2, func):
		pass
	
	def conv(obj1, obj2, func):
		n = obj1.len()
		if n != obj2.len():
			raise ValueError('length of both pboxes must be same')
		uxs = pbox.sortfunc(obj1.u.xs, obj2.u.xs, func)
		lxs = pbox.sortfunc(obj1.l.xs, obj2.l.xs, func)
		iu = slice(0, n**2, n)
		il = slice((n-1), n**2, n)
		return pbox(uxs[iu], lxs[il])
	
	def __add__(obj1, obj2):
		if (type(obj1) == pbox):
			if (type(obj2) == pbox):
				return pbox.conv(obj1, obj2, op.add)
			elif (type(obj2) == int) or (type(obj2) == float):
				uxs = np.add(obj1.u.xs, obj2)
				lxs = np.add(obj1.l.xs, obj2)
				return pbox(uxs, lxs)
		else:
			raise ValueError('this operation is not defined')
		
	def __radd__(self, obj1):
		if (type(obj1) == int) or (type(obj1) == float):
			return self + obj1
		else:
			raise ValueError('this operation is not defined')
	
	def __sub__(obj1, obj2):
		return obj1 + (-obj2)
	
	def __rsub__(self, obj1):
		return obj1 + (-self)
	
	def __mul__(obj1, obj2):
		if (obj1 < 0) or (obj2 < 0):
			raise ValueError('inputs can not be negative')
		else:	
			if (type(obj1) == pbox):
				if (type(obj2) == pbox):
					return pbox.conv(obj1, obj2, op.mul)
				elif (type(obj2) == int) or (type(obj2) == float):
					uxs = np.multiply(obj1.u.xs, obj2)
					lxs = np.multiply(obj1.l.xs, obj2)
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
		
	def __truediv__(obj1, obj2):
		if (obj1 <= 0) or (obj2 <= 0):
			raise ValueError('inputs must be strictly positive')
		else:
			return obj1 * (1/obj2)
		
	
	
	@staticmethod
		
	def sortfunc(xs1, xs2, func):
		pass
		n1 = np.size(xs1)
		n2 = np.size(xs2)
		ys1 = np.tile(xs1, n2)
		ys2 = np.repeat(xs2, n1)
		ys = func(ys1, ys2)
		return sorted(ys)