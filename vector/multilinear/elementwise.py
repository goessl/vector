from ..functional.elementwise import vechadamardmax
import numpy as np



__all__ = ('tenhadamard', 'tenhadamardtruediv',
           'tenhadamardfloordiv', 'tenhadamardmod', 'tenhadamarddivmod',
           'tenhadamardmin', 'tenhadamardmax')



def tenhadamard(*ts):
    r"""Return the elementwise product.
    
    $$
        \left((t_0)_i \cdot (t_1)_i \cdot \cdots\right)_i
    $$
    """
    ts = tuple(map(np.asarray, ts))
    shape = tuple(map(min, zip(*(t.shape for t in ts))))
    r = np.zeros(shape, dtype=np.result_type(*ts) if ts else object)
    slc = tuple(map(slice, shape)) + (...,)
    if ts:
        r = ts[0][*slc]
    for t in ts[1:]:
        r *= t[*slc]
    return r

def tenhadamardtruediv(s, t):
    r"""Return the elementwise true quotient.
    
    $$
        \left(\frac{s_i}{t_i}\right)_i
    $$
    """
    s, t = np.asarray(s), np.asarray(t)
    return np.divide(s, t[tuple(map(slice, s.shape)), ...])

def tenhadamardfloordiv(s, t):
    r"""Return the elementwise floor quotient.
    
    $$
        \left(\left\lfloor\frac{s_i}{t_i}\right\rfloor\right)_i
    $$
    """
    s, t = np.asarray(s), np.asarray(t)
    return np.floor_divide(s, t[tuple(map(slice, s.shape)), ...])

def tenhadamardmod(s, t):
    r"""Return the elementwise remainder.
    
    $$
        \left(s_i \bmod t_i\right)_i
    $$
    """
    s, t = np.asarray(s), np.asarray(t)
    return np.mod(s, t[tuple(map(slice, s.shape)), ...])

def tenhadamarddivmod(s, t):
    r"""Return the elementwise floor quotient and remainder.
    
    $$
        \left(\left\lfloor\frac{s_i}{t_i}\right\rfloor\right)_i, \ \left(s_i \bmod t_i\right)_i
    $$
    """
    s, t = np.asarray(s), np.asarray(t)
    return np.divmod(s, t[tuple(map(slice, s.shape)), ...])

def tenhadamardmin(*ts, key=None):
    r"""Return the elementwise minimum.
    
    $$
        \left(\min((t_0)_i, (t_1)_i, \cdots)\right)_i
    $$
    """
    ts = tuple(map(np.asarray, ts))
    shape = vechadamardmax(*(t.shape for t in ts))
    r = np.empty(shape, dtype=np.result_type(*ts) if ts else object)
    filled = np.zeros(shape, dtype=bool)
    if key is None:
        for t in ts:
            slc = tuple(map(slice, t.shape)) + (0,)*(r.ndim-t.ndim)
            r[slc] = np.where(filled[slc], np.minimum(r[slc], t), t)
            filled[slc] = True
    else:
        kcache = np.empty(shape, dtype=object)
        for t in ts:
            slc = tuple(map(slice, t.shape)) + (0,)*(r.ndim-t.ndim)
            it = np.nditer(
                [r[slc], t, filled[slc], kcache[slc]],
                flags = ['refs_ok'],
                op_flags = [['readwrite'], ['readonly'], ['readonly'], ['readwrite']]
            )
            for r_cell, t_cell, is_filled, k_cell in it:
                t_val = t_cell.item()
                t_key = key(t_val)
                if not bool(is_filled):
                    r_cell[...] = t_val
                    k_cell[...] = t_key
                else:
                    if t_key < k_cell.item():
                        r_cell[...] = t_val
                        k_cell[...] = t_key
            filled[slc] = True
    r[~filled] = 0
    return r

def tenhadamardmax(*ts, key=None):
    r"""Return the elementwise maximum.
    
    $$
        \left(\min((t_0)_i, (t_1)_i, \cdots)\right)_i
    $$
    """
    ts = tuple(map(np.asarray, ts))
    shape = vechadamardmax(*(t.shape for t in ts))
    r = np.empty(shape, dtype=np.result_type(*ts) if ts else object)
    filled = np.zeros(shape, dtype=bool)
    if key is None:
        for t in ts:
            slc = tuple(map(slice, t.shape)) + (0,)*(r.ndim-t.ndim)
            r[slc] = np.where(filled[slc], np.maximum(r[slc], t), t)
            filled[slc] = True
    else:
        kcache = np.empty(shape, dtype=object)
        for t in ts:
            slc = tuple(map(slice, t.shape)) + (0,)*(r.ndim-t.ndim)
            it = np.nditer(
                [r[slc], t, filled[slc], kcache[slc]],
                flags = ['refs_ok'],
                op_flags = [['readwrite'], ['readonly'], ['readonly'], ['readwrite']]
            )
            for r_cell, t_cell, is_filled, k_cell in it:
                t_val = t_cell.item()
                t_key = key(t_val)
                if not bool(is_filled):
                    r_cell[...] = t_val
                    k_cell[...] = t_key
                else:
                    if t_key > k_cell.item():
                        r_cell[...] = t_val
                        k_cell[...] = t_key
            filled[slc] = True
    r[~filled] = 0
    return r
