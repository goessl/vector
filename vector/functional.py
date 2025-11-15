from random import random, gauss
from itertools import count, zip_longest, tee
from operator import mul
from .lazy import vecleq, veclround, veclrshift, vecllshift
from .lazy import try_conjugate, veclconj, veclpos, veclneg, vecladd, vecladdc, veclsub, veclsubc, veclmul, vecltruediv, veclfloordiv, veclmod
from .lazy import veclhadamard, veclhadamardtruediv, veclhadamardfloordiv, veclhadamardmod, veclhadamarddivmod, veclhadamardmin, veclhadamardmax
from operationcounter import sumprod_default



__all__ = (#creation
           'veczero', 'vecbasis', 'vecbases', 'vecrand', 'vecrandn',
           #utility
           'veceq', 'vectrim', 'vecround', 'vecrshift', 'veclshift',
           #Hilbert space
           'vecconj', 'vecabs', 'vecabsq', 'vecdot', 'vecparallel',
           #vector space
           'vecpos', 'vecneg', 'vecadd', 'vecaddc', 'vecsub', 'vecsubc',
           'vecmul', 'vectruediv', 'vecfloordiv', 'vecmod', 'vecdivmod',
           #elementwise
           'vechadamard', 'vechadamardtruediv',
           'vechadamardfloordiv', 'vechadamardmod', 'vechadamarddivmod',
           'vechadamardmin', 'vechadamardmax')



#creation
veczero = ()
r"""Zero vector.

$$
    \vec{0} \qquad \mathbb{K}^0
$$
"""

def vecbasis(i, c=1, zero=0):
    r"""Return the `i`-th basis vector times `c`.
    
    $$
        c\vec{e}_i \qquad \mathbb{K}^{i+1}
    $$
    
    Returns a tuple with `i` many `zero`s followed by `c`.
    """
    return (zero,)*i + (c,)

def vecbases(start=0, c=1, zero=0):
    r"""Yield all basis vectors.
    
    $$
        \left(\vec{e}_n\right)_\mathbb{n\in\mathbb{N_0}} = \left(\vec{e}_0, \vec{e}_1, \vec{e}_2, \dots \right)
    $$
    
    See also
    --------
    - for single basis vector: [`vecbasis`][vector.functional.vecbasis]
    """
    for i in count(start=start):
        yield vecbasis(i, c=c, zero=zero)

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

def vecrandn(n, normed=True, mu=0, sigma=1, weights=None):
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
    return vectruediv(v, vecabs(v, weights)) if normed else v


#utility
def veceq(v, w, zero=0):
    r"""Return if two vectors are equal.
    
    $$
        \vec{v}=\vec{w} \qquad \mathbb{K}^m\times\mathbb{K}^n\to\mathbb{B}
    $$
    
    For two vectors of lengths $n$ & $m$ there will be at most
    
    - $\min\{n, m\}$ scalar comparisons (`eq`) &
    - $|n-m|$ scalar boolean evaluations (`bool`).
    """
    #return all(starmap(eq, zip_longest(v, w, fillvalue=zero)))
    return all(vecleq(v, w))

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
    return tuple(veclround(v, ndigits=ndigits))

def vecrshift(v, n, zero=0):
    r"""Pad `n` many `zero`s to the beginning of the vector.
    
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
    return tuple(veclrshift(v, n, zero=zero))

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
    return tuple(vecllshift(v, n))


#Hilbert space
def vecconj(v):
    r"""Return the elementwise complex conjugate.
    
    $$
        \vec{v}^* \qquad \mathbb{K}^n\to\mathbb{K}^n
    $$
    
    Trys to call a method `conjugate` on each element.
    If not found, simply keeps the element as is.
    
    For a vector of length $n$ there will be
    
    - $n$ scalar conjugations.
    """
    return tuple(veclconj(v))

def vecabs(v, weights=None, conjugate=False, zero=0):
    r"""Return the Euclidean/L2-norm.
    
    $$
        ||\vec{v}||_{\ell_{\mathbb{N}_0}^2}=\sqrt{\sum_iv_i^{(*)}v_i\omega_i} \qquad \mathbb{K}^n\to\mathbb{K}_0^+
    $$
    
    Returns the square root of [`vecabsq`][vector.functional.vecabsq].
    """
    #hypot(*v) doesn't work for complex
    #math.sqrt doesn't work for complex and cmath.sqrt always returns complex
    return vecabsq(v, weights=weights, conjugate=conjugate, zero=zero)**0.5

def vecabsq(v, weights=None, conjugate=False, zero=0):
    r"""Return the sum of absolute squares of the coefficients.
    
    $$
        ||\vec{v}||_{\ell_{\mathbb{N}_0}^2}^2=\sum_iv_i^{(*)}v_i\omega_i \qquad \mathbb{K}^n\to\mathbb{K}_0^+
    $$
    
    For a vector of length $n$ there will be
    
    - $n$ scalar conjugations (if selected) (`conjugate`),
    - $n$/$2n$ scalar multiplications without/with weights (`mul`) &
    - $\begin{cases}n-1&n\ge1\\0&n\le1\end{cases}$ scalar additions (`add`).
    
    Notes
    -----
    Reasons why it exists:
    
    - Occurs in math.
    - Most importantly: type independent because it doesn't use `sqrt`.
    
    References
    ----------
    - <https://docs.python.org/3/library/itertools.html#itertools-recipes>: `sum_of_squares`
    """
    vc, v = tee(v, 2)
    if conjugate:
        vc = veclconj(vc)
    
    if weights is None:
        return sumprod_default(vc, v, default=zero)
    else:
        return sumprod_default(map(mul, vc, v), weights, default=zero)

def vecdot(v, w, weights=None, conjugate=False, zero=0):
    r"""Return the inner product of two vectors.
    
    $$
        \left<\vec{v}\mid\vec{w}\right>_{\ell_{\mathbb{N}_0}^2}=\sum_iv_i^{(*)}w_i\omega_i \qquad \mathbb{K}^m\times\mathbb{K}^n\to\mathbb{K}
    $$
    
    For two vectors of lengths $n$ & $m$ there will be
    
    - $\min\{n, m\}$ scalar conjugations (if selected) (`conjugate`),
    - $\min\{n, m\}$/$2\min\{n, m\}$ scalar multiplications without/with weights (`mul`) &
    - $\begin{cases}\min\{n, m\}-1&n\ge1\land m\ge1\\0&n\le1\lor m\le1\end{cases}$ scalar additions (`add`).
    """
    if conjugate:
        v = veclconj(v)
    
    if weights is None:
        return sumprod_default(v, w, default=zero)
    else:
        return sumprod_default(map(mul, v, w), weights, default=zero)

def vecparallel(v, w, weights=None, conjugate=False, zero=0):
    r"""Return if two vectors are parallel.
    
    $$
        \vec{v}\parallel\vec{w} \qquad ||\vec{v}||\,||\vec{w}|| \overset{?}{=} |\vec{v}\vec{w}|^2 \qquad \mathbb{K}^m\times\mathbb{K}^n\to\mathbb{B}
    $$
    """
    #doesn't work for exhaustible iterables
    #return vecabsq(v)*vecabsq(w) == abs(vecdot(v, w))**2
    v2, w2, vw = zero, zero, zero
    if weights is None:
        for vi, wi in zip_longest(v, w, fillvalue=zero):
            vic, wic = (try_conjugate(vi), try_conjugate(wi)) if conjugate else (vi, wi)
            v2 += vic * vi
            w2 += wic * wi
            vw += vic * wi
    else:
        for vi, wi, o in zip_longest(v, w, weights, fillvalue=zero):
            vic, wic = (try_conjugate(vi), try_conjugate(wi)) if conjugate else (vi, wi)
            v2 += vic * vi * o
            w2 += wic * wi * o
            vw += vic * wi * o
    return v2 * w2 == try_conjugate(vw) * vw


#vector space
def vecpos(v):
    r"""Return the vector with the unary positive operator applied.
    
    $$
        +\vec{v} \qquad \mathbb{K}^n\to\mathbb{K}^n
    $$
    
    For a vector of length $n$ there will be
    
    - $n$ scalar unary plus operations (`pos`).
    """
    return tuple(veclpos(v))

def vecneg(v):
    r"""Return the vector with the unary negative operator applied.
    
    $$
        -\vec{v} \qquad \mathbb{K}^n\to\mathbb{K}^n
    $$
    
    For a vector of length $n$ there will be
    
    - $n$ scalar negations (`neg`).
    """
    return tuple(veclneg(v))

def vecadd(*vs):
    r"""Return the sum of vectors.
    
    $$
        \vec{v}_0+\vec{v}_1+\cdots \qquad \mathbb{K}^{n_0}\times\mathbb{K}^{n_1}\times\cdots\to\mathbb{K}^{\max_i n_i}
    $$
    
    For two vectors of lengths $n$ & $m$ there will be
    
    - $\min\{n, m\}$ scalar additions (`add`).
    """
    return tuple(vecladd(*vs))

def vecaddc(v, c, i=0, zero=0):
    r"""Return `v` with `c` added to the `i`-th coefficient.
    
    $$
        \vec{v}+c\vec{e}_i \qquad \mathbb{K}^n\to\mathbb{K}^{\max\{n, i\}}
    $$
    
    More efficient than `vecadd(v, vecbasis(i, c))`.
    
    There will be
    
    - one scalar addition (`add`) or one scalar identity (`pos`) 
    """
    return tuple(vecladdc(v, c, i=i, zero=zero))

def vecsub(v, w):
    r"""Return the difference of two vectors.
    
    $$
        \vec{v}-\vec{w} \qquad \mathbb{K}^m\times\mathbb{K}^n\to\mathbb{K}^{\max\{m, n\}}
    $$
    
    For two vectors of length $n$ & $m$ there will be
    
    - $\min\{n, m\}$ scalar subtractions (`-`) &
    - $\begin{cases}m-n&m\ge n\\0&m\le n\end{cases}$ negations.
    """
    return tuple(veclsub(v, w))

def vecsubc(v, c, i=0, zero=0):
    r"""Return `v` with `c` added to the `i`-th coefficient.
    
    $$
        \vec{v}-c\vec{e}_i \qquad \mathbb{K}^n\to\mathbb{K}^{\max\{n, i\}}
    $$
    
    More efficient than `vecsub(v, vecbasis(i, c))`.
    
    There will be
    
    - one scalar subtraction (`sub`) or one scalar negation (`neg`) 
    """
    return tuple(veclsubc(v, c, i=i, zero=zero))

def vecmul(a, v):
    r"""Return the product of a scalar and a vector.
    
    $$
        a\vec{v} \qquad \mathbb{K}\times\mathbb{K}^n\to\mathbb{K}^n
    $$
    
    For a vector of length $n$ there will be
    
    - $n$ scalar multiplications (`rmul`).
    """
    return tuple(veclmul(a, v))

def vectruediv(v, a):
    r"""Return the true division of a vector and a scalar.
    
    $$
        \frac{\vec{v}}{a} \qquad \mathbb{K}^n\times\mathbb{K}\to\mathbb{K}^n
    $$
    
    For a vector of length $n$ there will be
    
    - $n$ scalar true divisions (`truediv`).
    
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
    return tuple(vecltruediv(v, a))

def vecfloordiv(v, a):
    r"""Return the floor division of a vector and a scalar.
    
    $$
        \left(\left\lfloor\frac{v_i}{a}\right\rfloor\right)_i \qquad \mathbb{K}^n\times\mathbb{K}\to\mathbb{K}^n
    $$
    
    For a vector of length $n$ there will be
    
    - $n$ scalar floor divisions (`floordiv`).
    """
    return tuple(veclfloordiv(v, a))

def vecmod(v, a):
    r"""Return the elementwise mod of a vector and a scalar.
    
    $$
        \left(v_i \mod a\right)_i \qquad \mathbb{K}^n\times\mathbb{K}\to\mathbb{K}^n
    $$
    
    For a vector of length $n$ there will be
    
    - $n$ scalar modulos (`mod`).
    """
    return tuple(veclmod(v, a))

def vecdivmod(v, a):
    r"""Return the elementwise divmod of a vector and a scalar.
    
    $$
        \left(\left\lfloor\frac{v_i}{a}\right\rfloor\right)_i, \ \left(v_i \mod a\right)_i \qquad \mathbb{K}^n\times\mathbb{K}\to\mathbb{K}^n\times\mathbb{K}^n
    $$
    
    For a vector of length $n$ there will be
    
    - $n$ scalar divmods (`divmod`).
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
    
    For vectors of lengths $n_1, n_2, \dots, n_N$ there will be
    
    - $\begin{cases}(N-1)\min_in_i&N\ge1\land\min_in_i\ge1\\0&N\le1\lor\min_in_i=0\end{cases}$ scalar multiplications (`mul`).
    """
    return tuple(veclhadamard(*vs))

def vechadamardtruediv(v, w):
    r"""Return the elementwise true division of two vectors.
    
    $$
        \left(\frac{v_i}{w_i}\right)_i \qquad \mathbb{K}^m\times\mathbb{K}^n\to\mathbb{K}^m
    $$
    
    For two vectors of lengths $n$ & $m$ there will be
    
    - $n$ scalar true divisions (`truediv`).
    """
    return tuple(veclhadamardtruediv(v, w))

def vechadamardfloordiv(v, w):
    r"""Return the elementwise floor division of two vectors.
    
    $$
        \left(\left\lfloor\frac{v_i}{w_i}\right\rfloor\right)_i \qquad \mathbb{K}^m\times\mathbb{K}^n\to\mathbb{K}^m
    $$
    
    For two vectors of lengths $n$ & $m$ there will be
    
    - $n$ scalar floor divisions (`floordiv`).
    """
    return tuple(veclhadamardfloordiv(v, w))

def vechadamardmod(v, w):
    r"""Return the elementwise mod of two vectors.
    
    $$
        \left(v_i \mod w_i\right)_i \qquad \mathbb{K}^m\times\mathbb{K}^n\to\mathbb{K}^m
    $$
    
    For two vectors of lengths $n$ & $m$ there will be
    
    - $n$ scalar modulos (`mod`).
    """
    return tuple(veclhadamardmod(v, w))

def vechadamarddivmod(v, w):
    r"""Return the elementwise divmod of two vectors.
    
    $$
        \left(\left\lfloor\frac{v_i}{w_i}\right\rfloor\right)_i, \ \left(v_i \mod w_i\right)_i \qquad \mathbb{K}^n\times\mathbb{K}^m\to\mathbb{K}^n\times\mathbb{K}^n
    $$
    
    For two vectors of lengths $n$ & $m$ there will be
    
    - $n$ scalar divmods (`divmod`).
    """
    q, r = [], []
    for qi, ri in veclhadamarddivmod(v, w):
        q.append(qi)
        r.append(ri)
    return tuple(q), tuple(r)

def vechadamardmin(*vs, key=None):
    r"""Return the elementwise minimum of vectors.
    
    $$
        \left(\min((\vec{v}_0)_i,(\vec{v}_1)_i,\cdots)\right)_i \qquad \mathbb{K}^{n_0}\times\mathbb{K}^{n_1}\times\cdots\to\mathbb{K}^{\max_i n_i}
    $$
    
    For two vectors of lengths $n$ & $m$ there will be
    
    - $\min\{n, m\}$ comparisons (`lt`).
    """
    return tuple(veclhadamardmin(*vs, key=key))

def vechadamardmax(*vs, key=None):
    r"""Return the elementwise maximum of vectors.
    
    $$
        \left(\max((\vec{v}_0)_i,(\vec{v}_1)_i,\cdots)\right)_i \qquad \mathbb{K}^{n_0}\times\mathbb{K}^{n_1}\times\cdots\to\mathbb{K}^{\max_i n_i}
    $$
    
    For two vectors of lengths $n$ & $m$ there will be
    
    - $\min\{n, m\}$ comparisons (`gt`).
    """
    return tuple(veclhadamardmax(*vs, key=key))

def vechadamardminmax(*vs):
    pass
