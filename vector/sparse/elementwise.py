from operationcounter import MISSING, prod_default



__all__ = ('vecshadamard', 'vecshadamardtruediv',
           'vecshadamardfloordiv', 'vecshadamardmod', 'vecshadamarddivmod',
           'vecshadamardmin', 'vecshadamardmax')



def vecshadamard(*vs):
    r"""Return the elementwise product of vectors.
    
    $$
        \left((\vec{v}_0)_i\cdot(\vec{v}_1)_i\cdot\cdots\right)_i \qquad \mathbb{K}^{n_0}\times\mathbb{K}^{n_1}\times\cdots\to\mathbb{K}^{\min_i n_i}
    $$
    """
    r = {}
    if not vs:
        return r
    for k in set(vs[0].keys()).intersection(*(v.keys() for v in vs[1:])):
        r[k] = prod_default((v[k] for v in vs), initial=MISSING, default=MISSING)
    return r

def vecshadamardtruediv(v, w):
    r"""Return the elementwise true division of two vectors.
    
    $$
        \left(\frac{v_i}{w_i}\right)_i \qquad \mathbb{K}^m\times\mathbb{K}^n\to\mathbb{K}^m
    $$
    """
    r = {}
    for i, vi in v.items():
        try:
            wi = w[i]
        except KeyError:
            raise ZeroDivisionError
        r[i] = vi / wi
    return r

def vecshadamardfloordiv(v, w):
    r"""Return the elementwise floor division of two vectors.
    
    $$
        \left(\left\lfloor\frac{v_i}{w_i}\right\rfloor\right)_i \qquad \mathbb{K}^m\times\mathbb{K}^n\to\mathbb{K}^m
    $$
    """
    r = {}
    for i, vi in v.items():
        try:
            wi = w[i]
        except KeyError:
            raise ZeroDivisionError
        r[i] = vi // wi
    return r

def vecshadamardmod(v, w):
    r"""Return the elementwise mod of two vectors.
    
    $$
        \left(v_i \mod w_i\right)_i \qquad \mathbb{K}^m\times\mathbb{K}^n\to\mathbb{K}^m
    $$
    """
    r = {}
    for i, vi in v.items():
        try:
            wi = w[i]
        except KeyError:
            raise ZeroDivisionError
        r[i] = vi % wi
    return r

def vecshadamarddivmod(v, w):
    r"""Return the elementwise divmod of two vectors.
    
    $$
        \left(\left\lfloor\frac{v_i}{w_i}\right\rfloor\right)_i, \ \left(v_i \mod w_i\right)_i \qquad \mathbb{K}^n\times\mathbb{K}^m\to\mathbb{K}^n\times\mathbb{K}^n
    $$
    """
    q, r = {}, {}
    for i, vi in v.items():
        try:
            wi = w[i]
        except KeyError:
            raise ZeroDivisionError
        q[i], r[i] = divmod(vi, wi)
    return q, r

def vecshadamardmin(*vs, key=None):
    r"""Return the elementwise minimum of vectors.
    
    $$
        \left(\min((\vec{v}_0)_i,(\vec{v}_1)_i,\cdots)\right)_i \qquad \mathbb{K}^{n_0}\times\mathbb{K}^{n_1}\times\cdots\to\mathbb{K}^{\max_i n_i}
    $$
    """
    r = {}
    if not vs:
        return r
    for k in set(vs[0].keys()).union(*(v.keys() for v in vs[1:])):
        r[k] = min(v[k] for v in vs if k in v)
    return r

def vecshadamardmax(*vs, key=None):
    r"""Return the elementwise maximum of vectors.
    
    $$
        \left(\max((\vec{v}_0)_i,(\vec{v}_1)_i,\cdots)\right)_i \qquad \mathbb{K}^{n_0}\times\mathbb{K}^{n_1}\times\cdots\to\mathbb{K}^{\max_i n_i}
    $$
    """
    r = {}
    if not vs:
        return r
    for k in set(vs[0].keys()).union(*(v.keys() for v in vs[1:])):
        r[k] = max(v[k] for v in vs if k in v)
    return r
