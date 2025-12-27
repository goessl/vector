from operationcounter import MISSING, prod_default



__all__ = ('tenshadamard', 'tenshadamardtruediv',
           'tenshadamardfloordiv', 'tenshadamardmod', 'tenshadamarddivmod',
           'tenshadamardmin', 'tenshadamardmax')



def tenshadamard(*ts):
    r"""Return the elementwise product.
    
    $$
        \left((t_0)_i \cdot (t_1)_i \cdot \cdots\right)_i
    $$
    """
    r = {}
    if not ts:
        return r
    for i in set(ts[0].keys()).intersection(*(t.keys() for t in ts[1:])):
        r[i] = prod_default((t[i] for t in ts), initial=MISSING, default=MISSING)
    return r

def tenshadamardtruediv(s, t):
    r"""Return the elementwise true quotient.
    
    $$
        \left(\frac{s_i}{t_i}\right)_i
    $$
    """
    return {i:si/t[i] for i, si in s.items()}

def tenshadamardfloordiv(s, t):
    r"""Return the elementwise floor quotient.
    
    $$
        \left(\left\lfloor\frac{s_i}{t_i}\right\rfloor\right)_i
    $$
    """
    return {i:si//t[i] for i, si in s.items()}

def tenshadamardmod(s, t):
    r"""Return the elementwise remainder.
    
    $$
        \left(s_i \bmod t_i\right)_i
    $$
    """
    return {i:si%t[i] for i, si in s.items()}

def tenshadamarddivmod(s, t):
    r"""Return the elementwise floor quotient and remainder.
    
    $$
        \left(\left\lfloor\frac{s_i}{t_i}\right\rfloor\right)_i, \ \left(s_i \bmod t_i\right)_i
    $$
    """
    q, r = {}, {}
    for i, ti in t.items():
        si = s[i]
        q[i], r[i] = divmod(ti, si)
    return q, r

def tenshadamardmin(*ts, key=None):
    r"""Return the elementwise minimum.
    
    $$
        \left(\min((t_0)_i, (t_1)_i, \cdots)\right)_i
    $$
    """
    r = {}
    if not ts:
        return r
    for i in set(ts[0].keys()).union(*(t.keys() for t in ts[1:])):
        r[i] = min(t[i] for t in ts)
    return r

def tenshadamardmax(*ts, key=None):
    r"""Return the elementwise maximum.
    
    $$
        \left(\min((t_0)_i, (t_1)_i, \cdots)\right)_i
    $$
    """
    r = {}
    if not ts:
        return r
    for i in set(ts[0].keys()).union(*(t.keys() for t in ts[1:])):
        r[i] = max(t[i] for t in ts)
    return r
