from .creation import vecbasis, vecbases, vecrand, vecrandn
from .utility import veceq, vectrim, vecrshift, veclshift
from .hilbertspace import vecconj, vecabs, vecabsq
from .vectorspace import vecpos, vecneg, vecadd, vecaddc, vecsub, vecsubc, vecmul, vecrmul, vectruediv, vecfloordiv, vecmod, vecdivmod
from .elementwise import vechadamard, vechadamardtruediv, vechadamardfloordiv, vechadamardmod, vechadamarddivmod, vechadamardmin, vechadamardmax



class Vector:
    __slots__ = ('data',)
    
    
    
    @classmethod
    def basis(cls, i, c=1, zero=0):
        return cls(vecbasis(i, c=c, zero=zero))
    
    @classmethod
    def bases(cls, start=0, c=1, zero=0):
        for v in vecbases(start=start, c=c, zero=zero):
            yield cls(v)
    
    @classmethod
    def rand(cls, n):
        return cls(vecrand(n))
    
    @classmethod
    def randn(cls, n, normed=True, mu=0, sigma=1, weights=None):
        return cls(vecrandn(n, normed=normed, mu=mu, sigma=sigma, weights=weights))
    
    def __init__(self, data=()):
        #data is immutable, therefore use reference
        self.data = data
    
    
    
    #container
    def __bool__(self):
        return any(self)
    
    def __iter__(self):
        return iter(self.data)
    
    def __getitem__(self, key):
        if isinstance(key, slice):
            #enable stop>=len with zero padding
            if key.stop is not None:
                l = key.stop
            else:
                l = len(self)
            return type(self)(tuple(self[i] for i in range(*key.indices(l))))
        
        try:
            return self.data[key]
        except IndexError:
            return 0
    
    
    #utility
    def __len__(self):
        return len(self.data)
    
    def __eq__(self, other):
        return veceq(self, other)
    
    def trim(self, tol=None):
        return type(self)(vectrim(self, tol=tol))
    
    def __rshift__(self, other):
        return type(self)(vecrshift(self, other))
    
    def __lshift__(self, other):
        return type(self)(veclshift(self, other))
    
    
    #hilbertspace
    def conjugate(self):
        return type(self)(vecconj(self))
    
    def __abs__(self):
        return vecabs(self)
    
    def absq(self):
        return vecabsq(self)
    
    
    #vectorspace
    def __pos__(self):
        return type(self)(vecpos(self))
    
    def __neg__(self):
        return type(self)(vecneg(self))
    
    def __add__(self, other):
        return type(self)(vecadd(self, other))
    
    def addc(self, c, i=()):
        return type(self)(vecaddc(self, c, i=i))
    
    def __sub__(self, other):
        return type(self)(vecsub(self, other))
    
    def subc(self, c, i=()):
        return type(self)(vecsubc(self, c, i=i))
    
    def __mul__(self, other):
        return type(self)(vecmul(self, other))
    
    def __rmul__(self, other):
        return type(self)(vecrmul(other, self))
    
    def __truediv__(self, other):
        return type(self)(vectruediv(self, other))
    
    def __floordiv__(self, other):
        return type(self)(vecfloordiv(self, other))
    
    def __mod__(self, other):
        return type(self)(vecmod(self, other))
    
    def __divmod__(self, other):
        q, r = type(self)(), type(self)()
        q.data, r.data = vecdivmod(self, other)
        return q, r
    
    
    #elementwise
    def hadamard(self, *others):
        return type(self)(vechadamard(self, *others))
    
    def hadamardtruediv(self, other):
        return type(self)(vechadamardtruediv(self, other))
    
    def hadamardfloordiv(self, other):
        return type(self)(vechadamardfloordiv(self, other))
    
    def hadamardmod(self, other):
        return type(self)(vechadamardmod(self, other))
    
    def hadamarddivmod(self, other):
        return type(self)(vechadamarddivmod(self, other))
    
    def hadamardmin(self, *others):
        return type(self)(vechadamardmin(self, *others))
    
    def hadamardmax(self, *others):
        return type(self)(vechadamardmax(self, *others))
    
    
    #IO
    def __repr__(self):
        return f'{type(self).__name__}{self.data!r}'
