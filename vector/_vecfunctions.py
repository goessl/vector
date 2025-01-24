from math import prod, sumprod
from random import random, gauss
from itertools import starmap, zip_longest, repeat, tee
from operator import sub, mul, truediv, floordiv, mod, eq



__all__ = ['veczero', 'vecbasis', 'vecrand', 'vecrandn',
        'veceq', 'vectrim', 'vecround',
        'vecabsq', 'vecabs', 'vecdot',
        'vecpos', 'vecneg',
        'vecadd', 'vecsub', 'vecmul', 'vectruediv', 'vecfloordiv', 'vecmod',
        'vechadamard', 'vechadamardtruediv',
        'vechadamardfloordiv', 'vechadamardmod']



#creation stuff
veczero = ()
"""Zero vector."""

def vecbasis(i, c=1):
    """Return the `i`-th basis vector times `c`.
    
    The retured value is a tuple with `i` zeros followed by `c`.
    """
    return (0,)*i + (c,)

def vecrand(n):
    """Return a random vector of `n` uniform coefficients in `[0, 1[`."""
    return tuple(random() for _ in range(n))

def vecrandn(n, normed=True, mu=0, sigma=1):
    """Return a random vector of `n` normal distributed coefficients."""
    v = tuple(gauss(mu, sigma) for _ in range(n))
    return vectruediv(v, vecabs(v)) if normed else v


#utility stuff
def veceq(v, w):
    """Return if two vectors are equal."""
    return all(starmap(eq, zip_longest(v, w, fillvalue=0)))

def vectrim(v, tol=1e-9):
    """Remove all trailing near zero (abs(v_i)<=tol) coefficients."""
    #doesn't work for iterators
    #while v and abs(v[-1])<=tol:
    #    v = v[:-1]
    #return v
    r, t, it = [], [], iter(v)
    for x in v:
        t += [x]
        if abs(x)>tol:
            r += t
            t = []
    return tuple(r)

def vecround(v, ndigits=None):
    """Round all coefficients to the given precision."""
    return tuple(round(c, ndigits) for c in v)


#Hilbert space stuff
def vecabsq(v):
    """Return the sum of absolute squares of the coefficients."""
    #return sumprod(v, v) #no abs
    return sumprod(*tee(map(abs, v), 2))

def vecabs(v):
    """Return the Euclidean/L2-norm.
    
    Return the square root of `vecabsq`.
    """
    #hypot(*v) doesn't work for complex
    #math.sqrt doesn't work for complex and cmath.sqrt always returns complex
    #therefore use **0.5 instead of sqrt because it is type conserving
    return vecabsq(v)**0.5

def vecdot(v, w):
    """Return the inner product of two vectors without conjugation."""
    #unreadable:
    #return sumprod(v[:min(len(v), len(w))], w[:min(len(v), len(w))])
    #return sumprod(*zip(*zip(v, w))) would be more precise, but is bloat
    return sum(map(mul, v, w))


#vector space stuff
def vecpos(v):
    """Return the vector with the unary positive operator applied."""
    return tuple(map(pos, v))

def vecneg(v):
    """Return the vector with the unary negative operator applied."""
    return tuple(map(neg, v))

def vecadd(*vs):
    """Return the sum of vectors."""
    return tuple(map(sum, zip_longest(*vs, fillvalue=0)))

def vecsub(v, w):
    """Return the difference of two vectors."""
    return tuple(starmap(sub, zip_longest(v, w, fillvalue=0)))

def vecmul(a, v):
    """Return the product of a scalar and a vector."""
    return tuple(map(mul, repeat(a), v))

def vectruediv(v, a):
    """Return the true division of a vector and a scalar."""
    return tuple(map(truediv, v, repeat(a)))

def vecfloordiv(v, a):
    """Return the floor division of a vector and a scalar."""
    return tuple(map(floordiv, v, repeat(a)))

def vecmod(v, a):
    """Return the elementwise mod of a vector and a scalar."""
    return tuple(map(mod, v, repeat(a)))


#elementwise operations
def vechadamard(*vs):
    """Return the elementwise product of vectors."""
    return tuple(map(prod, zip(*vs)))

def vechadamardtruediv(v, w):
    """Return the elementwise true division of two vectors."""
    return tuple(map(truediv, v, w))

def vechadamardfloordiv(v, w):
    """Return the elementwise floor division of two vectors."""
    return tuple(map(floordiv, v, w))

def vechadamardmod(v, w):
    """Return the elementwise mod of two vectors."""
    return tuple(map(mod, v, w))
