from .creation import vecsbasis, vecsbases, vecsrand, vecsrandn
from .utility import vecslen, vecseq, vecstrim, vecsrshift, vecslshift
from .hilbert_space import vecsconj, vecsabs, vecsabsq
from .vector_space import vecspos, vecsneg, vecsadd, vecsaddc, vecssub, vecssubc, vecsmul, vecsrmul, vecstruediv, vecsfloordiv, vecsmod, vecsdivmod
from .elementwise import vecshadamard, vecshadamardtruediv, vecshadamardfloordiv, vecshadamardmod, vecshadamarddivmod, vecshadamardmin, vecshadamardmax



class VectorSparse:
    __slots__ = ('data',)
    
    
    
    @classmethod
    def basis(cls, i, c=1):
        return cls(vecsbasis(i, c=c))
    
    @classmethod
    def bases(cls, start=0, c=1):
        for v in vecsbases(start=start, c=c):
            yield cls(v)
    
    @classmethod
    def rand(cls, n):
        return cls(vecsrand(n))
    
    @classmethod
    def randn(cls, n, normed=True, mu=0, sigma=1, weights=None):
        return cls(vecsrandn(n, normed=normed, mu=mu, sigma=sigma, weights=weights))
    
    def __init__(self, data=None):
        self.data = dict(data) if data is not None else {}
    
    
    
    #container
    def __bool__(self):
        return any(self.values())
    
    def keys(self):
        return self.data.keys()
    
    def values(self):
        return self.data.values()
    
    def items(self):
        return self.data.items()
    
    def __iter__(self):
        return iter(self.data)
    
    def __contains__(self, key):
        return key in self.data
    
    def __getitem__(self, key):
        if isinstance(key, slice):
            #enable stop>=len with zero padding
            if key.stop is not None:
                l = key.stop
            else:
                l = len(self)
            return type(self)({i:self[i] for i in range(*key.indices(l))})
        
        return self.data.get(key, 0)
    
    
    #utility
    def __len__(self):
        return vecslen(self.data)
    
    def __eq__(self, other):
        return vecseq(self.data, other.data)
    
    def trim(self, tol=1e-9):
        return type(self)(vecstrim(self.data, tol=tol))
    
    def __rshift__(self, other):
        return type(self)(vecsrshift(self.data, other))
    
    def __lshift__(self, other):
        return type(self)(vecslshift(self.data, other))
    
    
    #hilbert_space
    def conjugate(self):
        return type(self)(vecsconj(self.data))
    
    def __abs__(self):
        return vecsabs(self.data)
    
    def absq(self):
        return vecsabsq(self.data)
    
    
    #vector_space
    def __pos__(self):
        return type(self)(vecspos(self.data))
    
    def __neg__(self):
        return type(self)(vecsneg(self.data))
    
    def __add__(self, other):
        return type(self)(vecsadd(self.data, other.data))
    
    def addc(self, c, i=()):
        return type(self)(vecsaddc(self.data, c, i=i))
    
    def __sub__(self, other):
        return type(self)(vecssub(self.data, other.data))
    
    def subc(self, c, i=()):
        return type(self)(vecssubc(self.data, c, i=i))
    
    def __mul__(self, other):
        return type(self)(vecsmul(self.data, other))
    
    def __rmul__(self, other):
        return type(self)(vecsrmul(other, self.data))
    
    def __truediv__(self, other):
        return type(self)(vecstruediv(self.data, other))
    
    def __floordiv__(self, other):
        return type(self)(vecsfloordiv(self.data, other))
    
    def __mod__(self, other):
        return type(self)(vecsmod(self.data, other))
    
    def __divmod__(self, other):
        q, r = type(self)(), type(self)()
        q.data, r.data = vecsdivmod(self.data, other)
        return q, r
    
    
    #elementwise
    def hadamard(self, *others):
        return type(self)(vecshadamard(self.data, *(other.data for other in others)))
    
    def hadamardtruediv(self, other):
        return type(self)(vecshadamardtruediv(self.data, other.data))
    
    def hadamardfloordiv(self, other):
        return type(self)(vecshadamardfloordiv(self.data, other.data))
    
    def hadamardmod(self, other):
        return type(self)(vecshadamardmod(self.data, other.data))
    
    def hadamarddivmod(self, other):
        return type(self)(vecshadamarddivmod(self.data, other.data))
    
    def hadamardmin(self, *others):
        return type(self)(vecshadamardmin(self.data, *(other.data for other in others)))
    
    def hadamardmax(self, *others):
        return type(self)(vecshadamardmax(self.data, *(other.data for other in others)))
    
    
    #IO
    def __repr__(self):
        return f'{type(self).__name__}{self.data!r}'
