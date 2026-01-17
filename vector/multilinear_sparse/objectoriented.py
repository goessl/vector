from .creation import tensbasis, tensrand, tensrandn
from .utility import tensrank, tensdim, tenseq, tenstrim, tensrshift, tenslshift
from .hilbert_space import tensconj
from .vector_space import tenspos, tensneg, tensadd, tensaddc, tenssub, tenssubc, tensmul, tensrmul, tenstruediv, tensfloordiv, tensmod, tensdivmod
from .elementwise import tenshadamard, tenshadamardtruediv, tenshadamardfloordiv, tenshadamardmod, tenshadamarddivmod, tenshadamardmin, tenshadamardmax
from ..functional.utility import vectrim



class TensorSparse:
    __slots__ = ('data',)
    
    
    
    @classmethod
    def basis(cls, *i, c=1):
        return cls(tensbasis(*i, c=c))
    
    @classmethod
    def rand(cls, *d):
        return cls(tensrand(*d))
    
    @classmethod
    def randn(cls, *d, mu=0, sigma=1):
        return cls(tensrandn(*d, mu=mu, sigma=sigma))
    
    def __init__(self, data=None):
        if data is None:
            self.data = {}
            return
        self.data = {vectrim(i):ti for i, ti in data.items()}
    
    
    
    #container
    def __bool__(self):
        return any(bool(ti) for ti in self.values())
    
    def __iter__(self):
        return iter(self.data)
    
    def __len__(self):
        return len(self.data)
    
    def __contains__(self, key):
        return vectrim(key) in self.data
    
    def __getitem__(self, key):
        #TODO: slicing
        return self.data.get(vectrim(key), 0)
    
    def keys(self):
        return self.data.keys()
    
    def values(self):
        return self.data.values()
    
    def items(self):
        return self.data.items()
    
    
    #utility
    def rank(self):
        return tensrank(self.data)
    
    def dim(self):
        return tensdim(self.data)
    
    def __eq__(self, other):
        return tenseq(self.data, other.data)
    
    def trim(self, tol=1e-9):
        r = type(self)()
        r.data = tenstrim(self.data, tol=tol)
        return r
    
    def __rshift__(self, other):
        r = type(self)()
        r.data = tensrshift(self.data, other)
        return r
    
    def __lshift__(self, other):
        r = type(self)()
        r.data = tenslshift(self.data, other)
        return r
    
    
    #hilbert_space
    def conjugate(self):
        r = type(self)()
        r.data = tensconj(self.data)
        return r
    
    
    #vector_space
    def __pos__(self):
        r = type(self)()
        r.data = tenspos(self.data)
        return r
    
    def __neg__(self):
        r = type(self)()
        r.data = tensneg(self.data)
        return r
    
    def __add__(self, other):
        r = type(self)()
        r.data = tensadd(self.data, other.data)
        return r
    
    def addc(self, c, i=()):
        r = type(self)()
        r.data = tensaddc(self.data, c, i=i)
        return r
    
    def __sub__(self, other):
        r = type(self)()
        r.data = tenssub(self.data, other.data)
        return r
    
    def subc(self, c, i=()):
        r = type(self)()
        r.data = tenssubc(self.data, c, i=i)
        return r
    
    def __mul__(self, other):
        r = type(self)()
        r.data = tensmul(self.data, other)
        return r
    
    def __rmul__(self, other):
        r = type(self)()
        r.data = tensrmul(other, self.data)
        return r
    
    def __truediv__(self, other):
        r = type(self)()
        r.data = tenstruediv(self.data, other)
        return r
    
    def __floordiv__(self, other):
        r = type(self)()
        r.data = tensfloordiv(self.data, other)
        return r
    
    def __mod__(self, other):
        r = type(self)()
        r.data = tensmod(self.data, other)
        return r
    
    def __divmod__(self, other):
        q, r = type(self)(), type(self)()
        q.data, r.data = tensdivmod(self.data, other)
        return q, r
    
    
    #elementwise
    def hadamard(self, *others):
        r = type(self)()
        r.data = tenshadamard(self.data, *(other.data for other in others))
        return r
    
    def hadamardtruediv(self, other):
        r = type(self)()
        r.data = tenshadamardtruediv(self.data, other.data)
        return r
    
    def hadamardfloordiv(self, other):
        r = type(self)()
        r.data = tenshadamardfloordiv(self.data, other.data)
        return r
    
    def hadamardmod(self, other):
        r = type(self)()
        r.data = tenshadamardmod(self.data, other.data)
        return r
    
    def hadamarddivmod(self, other):
        r = type(self)()
        r.data = tenshadamarddivmod(self.data, other.data)
        return r
    
    def hadamardmin(self, *others):
        r = type(self)()
        r.data = tenshadamardmin(self.data, *(other.data for other in others))
        return r
    
    def hadamardmax(self, *others):
        r = type(self)()
        r.data = tenshadamardmax(self.data, *(other.data for other in others))
        return r
    
    
    #IO
    def __repr__(self):
        return f'{type(self).__name__}{self.data!r}'
