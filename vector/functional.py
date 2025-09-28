from math import prod, sumprod, inf
from random import random, gauss
from itertools import starmap, zip_longest, repeat, tee, chain, islice
from operator import pos, neg, sub, mul, truediv, floordiv, mod, eq



__all__ = (#creation
           'veczero', 'vecbasis', 'vecrand', 'vecrandn',
           #utility
           'veceq', 'vectrim', 'vecround', 'vecrshift', 'veclshift',
           #Hilbert space
           'vecabsq',  'vecabs',  'vecdot', 'vecparallel',
           #vector space
           'vecpos', 'vecneg', 'vecadd', 'vecaddc', 'vecsub',
           'vecmul', 'vectruediv', 'vecfloordiv', 'vecmod', 'vecdivmod',
           #elementwise
           'vechadamard', 'vechadamardtruediv',
           'vechadamardfloordiv', 'vechadamardmod',
           'vechadamardmin', 'vechadamardmax')



#creation
veczero = ()
r"""Zero vector.

$$
    \vec{0} \qquad \mathbb{K}^0
$$
"""

def vecbasis(i, c=1):
    r"""Return the `i`-th basis vector times `c`.
    
    $$
        c\vec{e}_i \qquad \mathbb{K}^{i+1}
    $$
    
    Returns a tuple with `i` zeros followed by `c`.
    """
    return (0,)*i + (c,)

def vecrand(n):
    r"""Return a random vector of `n` uniform `float` coefficients in `[0, 1[`.
    
    $$
        \vec{v}\sim\mathcal{U}^n([0, 1[) \qquad \mathbb{K}^n
    $$
    
    Notes
    -----
    Naming like in `numpy.random`, because seems more concise
    (not `random` & `gauss` as in the stdlib).
    """
    return tuple(random() for _ in range(n))

def vecrandn(n, normed=True, mu=0, sigma=1):
    r"""Return a random vector of `n` normal distributed `float` coefficients.
    
    $$
        \vec{v}\sim\mathcal{N}^n(\mu, \sigma) \qquad \mathbb{K}^n
    $$
    
    Notes
    -----
    Naming like in `numpy.random`, because seems more concise
    (not `random` & `gauss` as in the stdlib).
    """
    v = tuple(gauss(mu, sigma) for _ in range(n))
    return vectruediv(v, vecabs(v)) if normed else v


#utility
def veceq(v, w):
    r"""Return if two vectors are equal.
    
    $$
        \vec{v}=\vec{w} \qquad \mathbb{K}^m\times\mathbb{K}^n\to\mathbb{B}
    $$
    """
    return all(starmap(eq, zip_longest(v, w, fillvalue=0)))

def vectrim(v, tol=1e-9):
    r"""Remove all trailing near zero (`abs(v_i)<=tol`) coefficients.
    
    $$
        \begin{pmatrix}
            v_0 \\
            \vdots \\
            v_m
        \end{pmatrix} \ \text{where} \ m=\max\{\, j\mid |v_{j-1}|>\text{tol}\,\}\cup\{-1\} \qquad \mathbb{K}^n\to\mathbb{K}^{\leq n}
    $$
    
    Notes
    -----
    - Cutting of elements that are `abs(vi)<=tol` instead of `abs(vi)<tol` to
    allow cutting of elements that are exactly zero by `trim(v, 0)` instead
    of `trim(v, sys.float_info.min)`.
    - `tol=1e-9` like in [PEP 485](https://peps.python.org/pep-0485/#defaults).
    """
    #doesn't work for iterators
    #while v and abs(v[-1])<=tol:
    #    v = v[:-1]
    #return v
    r, t = [], []
    for x in v:
        t.append(x)
        if abs(x)>tol:
            r.extend(t)
            t.clear()
    return tuple(r)

def vecround(v, ndigits=None):
    r"""Round all coefficients to the given precision.
    
    $$
        (\text{round}_\text{ndigits}(v_i))_i \qquad \mathbb{K}^n\to\mathbb{K}^n
    $$
    """
    return tuple(round(c, ndigits) for c in v)

def vecrshift(v, n, fill=0):
    r"""Pad `n` many `fill`s to the beginning of the vector.
    
    $$
        (v_{i-n})_i \qquad \begin{pmatrix}
            0 \\
            \vdots \\
            0 \\
            v_0 \\
            v_1 \\
            \vdots
        \end{pmatrix} \qquad \mathbb{K}^m\to\mathbb{K}^{m+n}
    $$
    """
    return tuple(chain((fill,)*n, v))

def veclshift(v, n):
    r"""Remove `n` many coefficients at the beginning of the vector.
    
    $$
        (v_{i+n})_i \qquad \begin{pmatrix}
            v_n \\
            v_{n+1} \\
            \vdots
        \end{pmatrix} \qquad \mathbb{K}^m\to\mathbb{K}^{\max\{m-n, 0\}}
    $$
    """
    return tuple(islice(v, n, None))


#Hilbert space
def vecabsq(v):
    r"""Return the sum of absolute squares of the coefficients.
    
    $$
        ||\vec{v}||_{\ell_{\mathbb{N}_0}^2}^2=\sum_i|v_i|^2 \qquad \mathbb{K}^n\to\mathbb{K}_0^+
    $$
    
    Notes
    -----
    Reasons why it exists:
    
    - Occurs in math.
    - Most importantly: type independent because it doesn't use `sqrt`.
    
    References
    ----------
    - <https://docs.python.org/3/library/itertools.html#itertools-recipes>: `sum_of_squares`
    """
    #return sumprod(v, v) #no abs
    return sumprod(*tee(map(abs, v), 2))

def vecabs(v):
    r"""Return the Euclidean/L2-norm.
    
    $$
        ||\vec{v}||_{\ell_{\mathbb{N}_0}^2}=\sqrt{\sum_i|v_i|^2} \qquad \mathbb{K}^n\to\mathbb{K}_0^+
    $$
    
    Returns the square root of [`vecabsq`][vector.functional.vecabsq].
    """
    #hypot(*v) doesn't work for complex
    #math.sqrt doesn't work for complex and cmath.sqrt always returns complex
    #therefore use **0.5 instead of sqrt because it is type conserving
    return vecabsq(v)**0.5

def vecdot(v, w):
    r"""Return the inner product of two vectors without conjugation.
    
    $$
        \left<\vec{v}\mid\vec{w}\right>_{\ell_{\mathbb{N}_0}^2}=\sum_iv_iw_i \qquad \mathbb{K}^m\times\mathbb{K}^n\to\mathbb{K}
    $$
    """
    #unreadable and doesn't work for generators
    #return sumprod(v[:min(len(v), len(w))], w[:min(len(v), len(w))])
    #return sumprod(*zip(*zip(v, w))) #would be more precise, but is bloat
    return sum(map(mul, v, w))

def vecparallel(v, w):
    r"""Return if two vectors are parallel.
    
    $$
        \vec{v}\parallel\vec{w} \qquad ||\vec{v}||\,||\vec{w}|| \overset{?}{=} |\vec{v}\vec{w}|^2 \qquad \mathbb{K}^m\times\mathbb{K}^n\to\mathbb{B}
    $$
    """
    #doesn't work for exhaustible iterables
    #return vecabsq(v)*vecabsq(w) == abs(vecdot(v, w))**2
    v2, w2, vw = 0, 0, 0
    for vi, wi in zip_longest(v, w, fillvalue=0):
        via, wia = abs(vi), abs(wi)
        v2 += via * via
        w2 += wia * wia
        vw += vi * wi
    vw = abs(vw)
    return v2 * w2 == vw * vw


#vector space
def vecpos(v):
    r"""Return the vector with the unary positive operator applied.
    
    $$
        +\vec{v} \qquad \mathbb{K}^n\to\mathbb{K}^n
    $$
    """
    return tuple(map(pos, v))

def vecneg(v):
    r"""Return the vector with the unary negative operator applied.
    
    $$
        -\vec{v} \qquad \mathbb{K}^n\to\mathbb{K}^n
    $$
    """
    return tuple(map(neg, v))

def vecadd(*vs):
    r"""Return the sum of vectors.
    
    $$
        \vec{v}_0+\vec{v}_1+\cdots \qquad \mathbb{K}^{n_0}\times\mathbb{K}^{n_1}\times\cdots\to\mathbb{K}^{\max_i n_i}
    $$
    """
    return tuple(map(sum, zip_longest(*vs, fillvalue=0)))

def vecaddc(v, c, i=0):
    r"""Return `v` with `c` added to the `i`-th coefficient.
    
    $$
        \vec{v}+c\vec{e}_i \qquad \mathbb{K}^n\to\mathbb{K}^{\max\{n, i\}}
    $$
    
    More efficient than `vecadd(v, vecbasis(i, c))`.
    """
    v = list(v)
    v.extend([0] * (i-len(v)+1))
    v[i] += c
    return tuple(v)

def vecsub(v, w):
    r"""Return the difference of two vectors.
    
    $$
        \vec{v}-\vec{w} \qquad \mathbb{K}^m\times\mathbb{K}^n\to\mathbb{K}^{\max\{m, n\}}
    $$
    """
    return tuple(starmap(sub, zip_longest(v, w, fillvalue=0)))

def vecmul(a, v):
    r"""Return the product of a scalar and a vector.
    
    $$
        a\vec{v} \qquad \mathbb{K}\times\mathbb{K}^n\to\mathbb{K}^n
    $$
    """
    return tuple(map(mul, repeat(a), v))

def vectruediv(v, a):
    r"""Return the true division of a vector and a scalar.
    
    $$
        \frac{\vec{v}}{a} \qquad \mathbb{K}^n\times\mathbb{K}\to\mathbb{K}^n
    $$
    
    Notes
    -----
    Why called `truediv` instead of `div`?
    
    - `div` would be more appropriate for an absolute clean mathematical
    implementation, that doesn't care about the language used. But the package
    might be used for pure integers/integer arithmetic, so both, `truediv`
    and `floordiv` operations have to be provided, and none should be
    privileged over the other by getting the universal `div` name.
    - `truediv`/`floordiv` is unambiguous, like Python `operator`s.
    """
    return tuple(map(truediv, v, repeat(a)))

def vecfloordiv(v, a):
    r"""Return the floor division of a vector and a scalar.
    
    $$
        \left(\left\lfloor\frac{v_i}{a}\right\rfloor\right)_i \qquad \mathbb{K}^n\times\mathbb{K}\to\mathbb{K}^n
    $$
    """
    return tuple(map(floordiv, v, repeat(a)))

def vecmod(v, a):
    r"""Return the elementwise mod of a vector and a scalar.
    
    $$
        \left(v_i \mod a\right)_i \qquad \mathbb{K}^n\times\mathbb{K}\to\mathbb{K}^n
    $$
    """
    return tuple(map(mod, v, repeat(a)))

def vecdivmod(v, a):
    r"""Return the elementwise divmod of a vector and a scalar.
    
    $$
        \left(\left\lfloor\frac{v_i}{a}\right\rfloor\right)_i, \ \left(v_i \mod a\right)_i \qquad \mathbb{K}^n\times\mathbb{K}\to\mathbb{K}^n\times\mathbb{K}^n
    $$
    """
    q, r = [], []
    for vi in v:
        qi, ri = divmod(vi, a)
        q.append(qi)
        r.append(ri)
    return tuple(q), tuple(r)


#elementwise
def vechadamard(*vs):
    r"""Return the elementwise product of vectors.
    
    $$
        \left((\vec{v}_0)_i\cdot(\vec{v}_1)_i\cdot\cdots\right)_i \qquad \mathbb{K}^{n_0}\times\mathbb{K}^{n_1}\times\cdots\to\mathbb{K}^{\min_i n_i}
    $$
    """
    return tuple(map(prod, zip(*vs)))

def vechadamardtruediv(v, w):
    r"""Return the elementwise true division of two vectors.
    
    $$
        \left(\frac{v_i}{w_i}\right)_i \qquad \mathbb{K}^m\times\mathbb{K}^n\to\mathbb{K}^m
    $$
    """
    return tuple(map(truediv, v, chain(w, repeat(0))))

def vechadamardfloordiv(v, w):
    r"""Return the elementwise floor division of two vectors.
    
    $$
        \left(\left\lfloor\frac{v_i}{w_i}\right\rfloor\right)_i \qquad \mathbb{K}^m\times\mathbb{K}^n\to\mathbb{K}^m
    $$
    """
    return tuple(map(floordiv, v, chain(w, repeat(0))))

def vechadamardmod(v, w):
    r"""Return the elementwise mod of two vectors.
    
    $$
        \left(v_i \mod w_i\right)_i \qquad \mathbb{K}^m\times\mathbb{K}^n\to\mathbb{K}^m
    $$
    """
    return tuple(map(mod, v, chain(w, repeat(0))))

def vechadamardmin(*vs):
    r"""Return the elementwise minimum of vectors.
    
    $$
        \left(\min((\vec{v}_0)_i,(\vec{v}_1)_i,\cdots)\right)_i \qquad \mathbb{K}^{n_0}\times\mathbb{K}^{n_1}\times\cdots\to\mathbb{K}^{\max_i n_i}
    $$
    """
    return tuple(map(min, zip_longest(*vs, fillvalue=inf)))

def vechadamardmax(*vs):
    r"""Return the elementwise maximum of vectors.
    
    $$
        \left(\max((\vec{v}_0)_i,(\vec{v}_1)_i,\cdots)\right)_i \qquad \mathbb{K}^{n_0}\times\mathbb{K}^{n_1}\times\cdots\to\mathbb{K}^{\max_i n_i}
    $$
    """
    return tuple(map(max, zip_longest(*vs, fillvalue=-inf)))

def vechadamardminmax(*vs):
    pass
