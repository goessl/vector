from ..dense.elementwise import vechadamardmax
import numpy as np



__all__ = ('tenpos', 'tenneg', 'tenadd', 'tenaddc', 'tensub', 'tensubc',
           'tenmul', 'tenrmul', 'tentruediv', 'tenfloordiv', 'tenmod', 'tendivmod')



def tenpos(t):
    """Return the identity.
    
    $$
        +t
    $$
    
    See also
    --------
    - wraps: [`numpy.positive`](https://numpy.org/doc/stable/reference/generated/numpy.positive.html)
    """
    return np.positive(t)

def tenneg(t):
    """Return the negation.
    
    $$
        -t
    $$
    
    See also
    --------
    - wraps: [`numpy.negative`](https://numpy.org/doc/stable/reference/generated/numpy.negative.html)
    """
    return np.negative(t)

def tenadd(*ts):
    r"""Return the sum.
    
    $$
        t_0 + t_1 + \cdots
    $$
    
    See also
    --------
    - for sum on a single coefficient: [`tenaddc`][vector.multilinear.vectorspace.tenaddc]
    """
    ts = tuple(map(np.asarray, ts))
    shape = vechadamardmax(*(t.shape for t in ts))
    r = np.zeros(shape, dtype=np.result_type(*ts) if ts else object)
    for t in ts:
        r[tuple(map(slice, t.shape)) + (0,)*(r.ndim-t.ndim)] += t
    return r

def tenaddc(t, c, i=(0,)):
    """Return the sum with a basis tensor.
    
    $$
        t+ce_i
    $$
    
    More efficient than `tenadd(t, tenbasis(i, c))`.
    
    See also
    --------
    - for sum on more coefficients: [`tenadd`][vector.multilinear.vectorspace.tenadd]
    """
    t = np.asarray(t)
    while t.ndim < len(i):
        t = np.expand_dims(t, axis=-1)
    t = np.pad(t, tuple((0, max(ii-s+1, 0)) for s, ii in zip(t.shape, i)))
    t[i + (0,)*(len(i)-t.ndim)] += c
    return t

def tensub(s, t):
    """Return the difference.
    
    $$
        s - t
    $$
    
    See also
    --------
    - for difference on a single coefficient: [`tensubc`][vector.multilinear.vectorspace.tensubc]
    """
    s, t = np.asarray(s), np.asarray(t)
    shape = vechadamardmax(s.shape, t.shape)
    r = np.zeros(shape, dtype=np.result_type(s, t))
    r[tuple(map(slice, s.shape)) + (0,)*(r.ndim-s.ndim)] = s
    r[tuple(map(slice, t.shape)) + (0,)*(r.ndim-t.ndim)] -= t
    return r

def tensubc(t, c, i=(0,)):
    """Return the difference with a basis tensor.
    
    $$
        t-ce_i
    $$
    
    More efficient than `tensub(t, tenbasis(i, c))`.
    
    See also
    --------
    - for difference on more coefficients: [`tensub`][vector.multilinear.vectorspace.tensub]
    """
    t = np.asarray(t)
    while t.ndim < len(i):
        t = np.expand_dims(t, axis=-1)
    t = np.pad(t, tuple((0, max(ii-s+1, 0)) for s, ii in zip(t.shape, i)))
    t[i + (0,)*(len(i)-t.ndim)] -= c
    return t

def tenmul(t, a):
    """Return the product.
    
    $$
        ta
    $$
    
    See also
    --------
    - wraps: [`numpy.multiply`](https://numpy.org/doc/stable/reference/generated/numpy.multiply.html)
    """
    return np.multiply(t, a)

def tenrmul(a, t):
    """Return the product.
    
    $$
        at
    $$
    
    See also
    --------
    - wraps: [`numpy.multiply`](https://numpy.org/doc/stable/reference/generated/numpy.multiply.html)
    """
    return np.multiply(a, t)

def tentruediv(t, a):
    r"""Return the true quotient.
    
    $$
        \frac{t}{a}
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
    
    See also
    --------
    - wraps: [`numpy.divide`](https://numpy.org/doc/stable/reference/generated/numpy.divide.html)
    """
    return np.divide(t, a)

def tenfloordiv(t, a):
    r"""Return the floor quotient.
    
    $$
        \left\lfloor\frac{t}{a}\right\rfloor
    $$
    
    See also
    --------
    - wraps: [`numpy.floor_divide`](https://numpy.org/doc/stable/reference/generated/numpy.floor_divide.html)
    """
    return np.floor_divide(t, a)

def tenmod(t, a):
    r"""Return the remainder.
    
    $$
        t \bmod a
    $$
    
    See also
    --------
    - wraps: [`numpy.mod`](https://numpy.org/doc/stable/reference/generated/numpy.mod.html)
    """
    return np.mod(t, a)

def tendivmod(t, a):
    r"""Return the floor quotient and remainder
    
    $$
        \left\lfloor\frac{t}{a}\right\rfloor, \ \left(t \bmod a\right)
    $$
    
    See also
    --------
    - wraps: [`numpy.divmod`](https://numpy.org/doc/stable/reference/generated/numpy.divmod.html)
    """
    return np.divmod(t, a)
