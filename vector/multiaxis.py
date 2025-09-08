import numpy as np
from .functional import vechadamardmax



__all__ = (#creation
           'tenzero', 'tenbasis', 'tenrand', 'tenrandn',
           #utility
           'tenrank', 'tendim', 'tentrim', 'tenround',
           #vector space
           'tenpos', 'tenneg', 'tenadd', 'tenaddc', 'tensub',
           'tenmul', 'tentruediv', 'tenfloordiv', 'tenmod', 'tendivmod',
           #elementwise
           'tenhadamard', 'tenhadamardtruediv',
           'tenhadamardfloordiv', 'tenhadamardmod')



#creation
tenzero = np.zeros((), dtype=object)
"""Zero tensor.

$$
    0
$$

See also
--------
[`veczero`][vector.functional.veczero]
"""
tenzero.flags.writeable = False

def tenbasis(i, c=1):
    """Return the `i`-th basis tensor times `c`.
    
    $$
        ce_i
    $$
    
    Returns a `numpy.ndarray` with `i+1` zeros in each direction and a `c` in
    the outer corner.
    
    See also
    --------
    [`vecbasis`][vector.functional.vecbasis]
    """
    t = np.zeros(np.add(i, 1), dtype=np.result_type(c))
    t[i] = c #dont unpack i, it might be a scalar
    return t

def tenrand(*d):
    r"""Return a random tensor of `d` uniform coefficients in `[0, 1[`.
    
    $$
        t \sim \mathcal{U}^d([0, 1[)
    $$
    
    `d` may be multiple dimensions.
    
    See also
    --------
    [`numpy.random.rand`](https://numpy.org/doc/stable/reference/generated/numpy.random.rand.html),
    [`vecrand`][vector.functional.vecrand]
    """
    return np.random.rand(*d)

def tenrandn(*d):
    r"""Return a random tensor of `d` normal distributed coefficients.
    
    $$
        t \sim \mathcal{N}^d
    $$
    
    `d` may be multiple dimensions.
    
    See also
    --------
    [`numpy.random.randn`](https://numpy.org/doc/stable/reference/generated/numpy.random.randn.html),
    [`vecrandn`][vector.functional.vecrandn]
    """
    return np.random.randn(*d)


#utility
def tenrank(t):
    r"""Return the rank of a tensor.
    
    $$
        \text{rank}\,t
    $$
    
    See also
    --------
    [`numpy.ndarray.ndim`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.ndim.html)
    """
    return np.asarray(t).ndim

def tendim(t):
    r"""Return the dimensionalities of a tensor.
    
    $$
        \dim t
    $$
    
    See also
    --------
    [`numpy.ndarray.shape`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.shape.html)
    """
    return np.asarray(t).shape

def tentrim(t, tol=1e-9):
    """Remove all trailing near zero (`abs(v_i)<=tol`) coefficients.
    
    See also
    --------
    [`vectrim`][vector.functional.vectrim]
    """
    t = np.asarray(t)
    for d in range(t.ndim): #reduce dimension
        i = (slice(None, None, None),)*d + (-1,) + (...,)
        while t.shape[d]>0 and np.all(np.abs(t[*i])<=tol):
            t = t[(slice(None, None, None),)*d + (slice(0, -1),) + (...,)]
    while t.shape and t.shape[-1] == 1: #reduce rank
        t = t[..., 0]
    return t

def tenround(t, ndigits=0):
    r"""Round all coefficients to the given precision.
    
    $$
        (\text{round}_\text{ndigits}(v_i))_i
    $$
    
    See also
    --------
    [`numpy.round`](https://numpy.org/doc/stable/reference/generated/numpy.round.html),
    [`vecround`][vector.functional.vecround]
    """
    return np.round(t, ndigits)


#vector space
def tenpos(t):
    """Return the tensor with the unary positive operator applied.
    
    $$
        +t
    $$
    
    See also
    --------
    [`numpy.ndarray.__pos__`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__pos__.html),
    [`vecpos`][vector.functional.vecpos]
    """
    return +np.asarray(t)

def tenneg(t):
    """Return the tensor with the unary negative operator applied.
    
    $$
        -t
    $$
    
    See also
    --------
    [`numpy.ndarray.__neg__`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__neg__.html),
    [`vecneg`][vector.functional.vecneg]
    """
    return -np.asarray(t)

def tenadd(*ts):
    r"""Return the sum of tensors.
    
    $$
        t_0 + t_1 + \cdots
    $$
    
    See also
    --------
    [`vecadd`][vector.functional.vecadd]
    """
    ts = tuple(map(np.asarray, ts))
    shape = vechadamardmax(*(t.shape for t in ts))
    r = np.zeros(shape, dtype=np.result_type(*ts) if ts else object)
    for t in ts:
        r[tuple(map(slice, t.shape)) + (0,)*(r.ndim-t.ndim)] += t
    return r

def tenaddc(t, c, i=(0,)):
    r"""Return `t` with `c` added to the `i`-th coefficient.
    
    More efficient than `tenadd(t, tenbasis(i, c))`.
    
    See also
    --------
    [`vecaddc`][vector.functional.vecaddc]
    """
    t = np.asarray(t)
    while t.ndim < len(i):
        t = np.expand_dims(t, axis=-1)
    t = np.pad(t, tuple((0, max(ii-s+1, 0)) for s, ii in zip(t.shape, i)))
    t[i + (0,)*(len(i)-t.ndim)] += c
    return t

def tensub(s, t):
    """Return the difference of two tensors.
    
    $$
        s - t
    $$
    
    See also
    --------
    [`vecsub`][vector.functional.vecsub]
    """
    s, t = np.asarray(s), np.asarray(t)
    shape = vechadamardmax(s.shape, t.shape)
    r = np.zeros(shape, dtype=np.result_type(s, t))
    r[tuple(map(slice, s.shape)) + (0,)*(r.ndim-s.ndim)] = s
    r[tuple(map(slice, t.shape)) + (0,)*(r.ndim-t.ndim)] -= t
    return r

def tenmul(a, t):
    """Return the product of a scalar and a tensor.
    
    $$
        a t
    $$
    
    See also
    --------
    [`numpy.ndarray.__mul__`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__mul__.html),
    [`vecmul`][vector.functional.vecmul]
    """
    return a * np.asarray(t)

def tentruediv(t, a):
    r"""Return the true division of a tensor and a scalar.
    
    $$
        \frac{t}{a}
    $$
    
    See also
    --------
    [`numpy.ndarray.__truediv__`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__truediv__.html),
    [`vectruediv`][vector.functional.vectruediv]
    """
    return np.asarray(t) / a

def tenfloordiv(t, a):
    r"""Return the floor division of a tensor and a scalar.
    
    $$
        \left(\left\lfloor\frac{t_i}{a}\right\rfloor\right)_i
    $$
    
    See also
    --------
    [`numpy.ndarray.__floordiv__`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__floordiv__.html),
    [`vecfloordiv`][vector.functional.vecfloordiv]
    """
    return np.asarray(t) // a

def tenmod(t, a):
    r"""Return the elementwise mod of a tensor and a scalar.
    
    $$
        \left(t_i \mod a\right)_i
    $$
    
    See also
    --------
    [`numpy.ndarray.__mod__`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__mod__.html),
    [`vecmod`][vector.functional.vecmod]
    """
    return np.asarray(t) % a

def tendivmod(t, a):
    r"""Return the elementwise divmod of a tensor and a scalar.
    
    $$
        \left(\left\lfloor\frac{t_i}{a}\right\rfloor\right)_i, \ \left(t_i \mod a\right)_i
    $$
    
    See also
    --------
    [`numpy.ndarray.__divmod__`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__divmod__.html),
    [`vecdivmod`][vector.functional.vecdivmod]
    """
    return divmod(np.asarray(t), a)


#elementwise
def tenhadamard(*ts):
    r"""Return the elementwise product of tensors.
    
    $$
        \left((t_0)_i \cdot (t_1)_i \cdot \cdots\right)_i
    $$
    
    See also
    --------
    [`vechadamard`][vector.functional.vechadamard]
    """
    ts = tuple(map(np.asarray, ts))
    shape = tuple(map(min, zip(*(t.shape for t in ts))))
    r = np.zeros(shape, dtype=np.result_type(*ts) if ts else object)
    if ts:
        r = ts[0][tuple(map(slice, shape)), ...]
    for t in ts[1:]:
        r *= t[tuple(map(slice, shape)), ...]
    return r

def tenhadamardtruediv(s, t):
    r"""Return the elementwise true division of two tensors.
    
    $$
        \left(\frac{s_i}{t_i}\right)_i
    $$
    
    See also
    --------
    [`vechadamardtruediv`][vector.functional.vechadamardtruediv]
    """
    s, t = np.asarray(s), np.asarray(t)
    r = np.zeros(s.shape, dtype=np.result_type(s, t))
    r = s[tuple(map(slice, r.shape)), ...]
    r /= t[tuple(map(slice, r.shape)), ...]
    return r

def tenhadamardfloordiv(s, t):
    r"""Return the elementwise floor division of two tensors.
    
    $$
        \left(\left\lfloor\frac{s_i}{t_i}\right\rfloor\right)_i
    $$
    
    See also
    --------
    [`vechadamardfloordiv`][vector.functional.vechadamardfloordiv]
    """
    s, t = np.asarray(s), np.asarray(t)
    r = np.zeros(s.shape, dtype=np.result_type(s, t))
    r = s[tuple(map(slice, r.shape)), ...]
    r //= t[tuple(map(slice, r.shape)), ...]
    return r

def tenhadamardmod(s, t):
    r"""Return the elementwise mod of two tensors.
    
    $$
        \left(s_i \mod t_i\right)_i
    $$
    
    See also
    --------
    [`vechadamardmod`][vector.functional.vechadamardmod]
    """
    s, t = np.asarray(s), np.asarray(t)
    r = np.zeros(s.shape, dtype=np.result_type(s, t))
    r = s[tuple(map(slice, r.shape)), ...]
    r %= t[tuple(map(slice, r.shape)), ...]
    return r
