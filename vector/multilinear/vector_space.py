from ..functional import vechadamardmax
import numpy as np



__all__ = ('tenpos', 'tenneg', 'tenadd', 'tenaddc', 'tensub', 'tensubc',
           'tenmul', 'tentruediv', 'tenfloordiv', 'tenmod', 'tendivmod')



def tenpos(t):
    """Return the tensor with the unary positive operator applied.
    
    $$
        +t
    $$
    
    See also
    --------
    - one-dimensional: [`vecpos`][vector.functional.vector_space.vecpos]
    - wraps: [`numpy.positive`](https://numpy.org/doc/stable/reference/generated/numpy.positive.html)
    """
    return np.positive(t)

def tenneg(t):
    """Return the tensor with the unary negative operator applied.
    
    $$
        -t
    $$
    
    See also
    --------
    - one-dimensional: [`vecneg`][vector.functional.vector_space.vecneg]
    - wraps: [`numpy.negative`](https://numpy.org/doc/stable/reference/generated/numpy.negative.html)
    """
    return np.negative(t)

def tenadd(*ts):
    r"""Return the sum of tensors.
    
    $$
        t_0 + t_1 + \cdots
    $$
    
    See also
    --------
    - one-dimensional: [`vecadd`][vector.functional.vector_space.vecadd]
    """
    ts = tuple(map(np.asarray, ts))
    shape = vechadamardmax(*(t.shape for t in ts))
    r = np.zeros(shape, dtype=np.result_type(*ts) if ts else object)
    for t in ts:
        r[tuple(map(slice, t.shape)) + (0,)*(r.ndim-t.ndim)] += t
    return r

def tenaddc(t, c, i=(0,)):
    """Return `t` with `c` added to the `i`-th coefficient.
    
    $$
        t+ce_i
    $$
    
    More efficient than `tenadd(t, tenbasis(i, c))`.
    
    See also
    --------
    - one-dimensional: [`vecaddc`][vector.functional.vector_space.vecaddc]
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
    - one-dimensional: [`vecsub`][vector.functional.vector_space.vecsub]
    """
    s, t = np.asarray(s), np.asarray(t)
    shape = vechadamardmax(s.shape, t.shape)
    r = np.zeros(shape, dtype=np.result_type(s, t))
    r[tuple(map(slice, s.shape)) + (0,)*(r.ndim-s.ndim)] = s
    r[tuple(map(slice, t.shape)) + (0,)*(r.ndim-t.ndim)] -= t
    return r

def tensubc(t, c, i=(0,)):
    """Return `t` with `c` subtracted from the `i`-th coefficient.
    
    $$
        t-ce_i
    $$
    
    More efficient than `tensub(t, tenbasis(i, c))`.
    
    See also
    --------
    - one-dimensional: [`vecsubc`][vector.functional.vector_space.vecsubc]
    """
    t = np.asarray(t)
    while t.ndim < len(i):
        t = np.expand_dims(t, axis=-1)
    t = np.pad(t, tuple((0, max(ii-s+1, 0)) for s, ii in zip(t.shape, i)))
    t[i + (0,)*(len(i)-t.ndim)] -= c
    return t

def tenmul(a, t):
    """Return the product of a scalar and a tensor.
    
    $$
        at
    $$
    
    See also
    --------
    - one-dimensional: [`vecmul`][vector.functional.vector_space.vecmul]
    - wraps: [`numpy.multiply`](https://numpy.org/doc/stable/reference/generated/numpy.multiply.html)
    """
    return np.multiply(a, t)

def tentruediv(t, a):
    r"""Return the true division of a tensor and a scalar.
    
    $$
        \frac{t}{a}
    $$
    
    See also
    --------
    - one-dimensional: [`vectruediv`][vector.functional.vector_space.vectruediv]
    - wraps: [`numpy.divide`](https://numpy.org/doc/stable/reference/generated/numpy.divide.html)
    """
    return np.divide(t, a)

def tenfloordiv(t, a):
    r"""Return the floor division of a tensor and a scalar.
    
    $$
        \left(\left\lfloor\frac{t_i}{a}\right\rfloor\right)_i
    $$
    
    See also
    --------
    - one-dimensional: [`vecfloordiv`][vector.functional.vector_space.vecfloordiv]
    - wraps: [`numpy.floor_divide`](https://numpy.org/doc/stable/reference/generated/numpy.floor_divide.html)
    """
    return np.floor_divide(t, a)

def tenmod(t, a):
    r"""Return the elementwise mod of a tensor and a scalar.
    
    $$
        \left(t_i \mod a\right)_i
    $$
    
    See also
    --------
    - one-dimensional: [`vecmod`][vector.functional.vector_space.vecmod]
    - wraps: [`numpy.mod`](https://numpy.org/doc/stable/reference/generated/numpy.mod.html)
    """
    return np.mod(t, a)

def tendivmod(t, a):
    r"""Return the elementwise divmod of a tensor and a scalar.
    
    $$
        \left(\left\lfloor\frac{t_i}{a}\right\rfloor\right)_i, \ \left(t_i \mod a\right)_i
    $$
    
    See also
    --------
    - one-dimensional: [`vecdivmod`][vector.functional.vector_space.vecdivmod]
    - wraps: [`numpy.divmod`](https://numpy.org/doc/stable/reference/generated/numpy.divmod.html)
    """
    return np.divmod(t, a)
