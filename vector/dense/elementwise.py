from ..lazy import veclhadamard, veclhadamardtruediv, veclhadamardfloordiv, veclhadamardmod, veclhadamarddivmod, veclhadamardmin, veclhadamardmax



__all__ = ('vechadamard', 'vechadamardtruediv',
           'vechadamardfloordiv', 'vechadamardmod', 'vechadamarddivmod',
           'vechadamardmin', 'vechadamardmax')



def vechadamard(*vs):
    r"""Return the elementwise product.
    
    $$
        \left((\vec{v}_0)_i\cdot(\vec{v}_1)_i\cdot\cdots\right)_i \qquad \mathbb{K}^{n_0}\times\mathbb{K}^{n_1}\times\cdots\to\mathbb{K}^{\min_i n_i}
    $$
    
    Complexity
    ----------
    For vectors of lengths $n_1, n_2, \dots, n_N$ there will be
    
    - $\begin{cases}(N-1)\min_in_i&N\ge1\land\min_in_i\ge1\\0&N\le1\lor\min_in_i=0\end{cases}$ scalar multiplications (`mul`).
    """
    return tuple(veclhadamard(*vs))

def vechadamardtruediv(v, w):
    r"""Return the elementwise true quotient.
    
    $$
        \left(\frac{v_i}{w_i}\right)_i \qquad \mathbb{K}^m\times\mathbb{K}^n\to\mathbb{K}^m
    $$
    
    Complexity
    ----------
    For two vectors of lengths $n$ & $m$ there will be
    
    - $n$ scalar true divisions (`truediv`).
    """
    return tuple(veclhadamardtruediv(v, w))

def vechadamardfloordiv(v, w):
    r"""Return the elementwise floor quotient.
    
    $$
        \left(\left\lfloor\frac{v_i}{w_i}\right\rfloor\right)_i \qquad \mathbb{K}^m\times\mathbb{K}^n\to\mathbb{K}^m
    $$
    
    Complexity
    ----------
    For two vectors of lengths $n$ & $m$ there will be
    
    - $n$ scalar floor divisions (`floordiv`).
    """
    return tuple(veclhadamardfloordiv(v, w))

def vechadamardmod(v, w):
    r"""Return the elementwise remainder.
    
    $$
        \left(v_i \bmod w_i\right)_i \qquad \mathbb{K}^m\times\mathbb{K}^n\to\mathbb{K}^m
    $$
    
    Complexity
    ----------
    For two vectors of lengths $n$ & $m$ there will be
    
    - $n$ scalar modulos (`mod`).
    """
    return tuple(veclhadamardmod(v, w))

def vechadamarddivmod(v, w):
    r"""Return the elementwise floor quotient and remainder.
    
    $$
        \left(\left\lfloor\frac{v_i}{w_i}\right\rfloor\right)_i, \ \left(v_i \bmod w_i\right)_i \qquad \mathbb{K}^n\times\mathbb{K}^m\to\mathbb{K}^n\times\mathbb{K}^n
    $$
    
    Complexity
    ----------
    For two vectors of lengths $n$ & $m$ there will be
    
    - $n$ scalar divmods (`divmod`).
    """
    q, r = [], []
    for qi, ri in veclhadamarddivmod(v, w):
        q.append(qi)
        r.append(ri)
    return tuple(q), tuple(r)

def vechadamardmin(*vs, key=None):
    r"""Return the elementwise minimum.
    
    $$
        \left(\min((\vec{v}_0)_i,(\vec{v}_1)_i,\cdots)\right)_i \qquad \mathbb{K}^{n_0}\times\mathbb{K}^{n_1}\times\cdots\to\mathbb{K}^{\max_i n_i}
    $$
    
    Complexity
    ----------
    For two vectors of lengths $n$ & $m$ there will be
    
    - $\min\{n, m\}$ comparisons (`lt`).
    """
    return tuple(veclhadamardmin(*vs, key=key))

def vechadamardmax(*vs, key=None):
    r"""Return the elementwise maximum.
    
    $$
        \left(\max((\vec{v}_0)_i,(\vec{v}_1)_i,\cdots)\right)_i \qquad \mathbb{K}^{n_0}\times\mathbb{K}^{n_1}\times\cdots\to\mathbb{K}^{\max_i n_i}
    $$
    
    Complexity
    ----------
    For two vectors of lengths $n$ & $m$ there will be
    
    - $\min\{n, m\}$ comparisons (`gt`).
    """
    return tuple(veclhadamardmax(*vs, key=key))
