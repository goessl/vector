__all__ = ('vecihadamard', 'vecihadamardtruediv',
           'vecihadamardfloordiv', 'vecihadamardmod')



def vecihadamard(v, *ws):
    r"""Return the elementwise product.
    
    $$
        \left((\vec{v})_i\cdot(\vec{w}_0)_i\cdot(\vec{w}_1)_i\cdot\cdots\right)_i \qquad \mathbb{K}^{n_0}\times\mathbb{K}^{n_1}\times\cdots\to\mathbb{K}^{\min_i n_i}
    $$
    """
    if ws:
        del v[min(len(w) for w in ws):]
        for w in ws:
            for i, wi in enumerate(w[:len(v)]):
                v[i] *= wi
    return v

def vecihadamardtruediv(v, w):
    r"""Return the elementwise true quotient.
    
    $$
        \left(\frac{v_i}{w_i}\right)_i \qquad \mathbb{K}^m\times\mathbb{K}^n\to\mathbb{K}^m
    $$
    """
    for i in range(len(v)):
        v[i] /= w[i]
    return v

def vecihadamardfloordiv(v, w):
    r"""Return the elementwise floor quotient.
    
    $$
        \left(\left\lfloor\frac{v_i}{w_i}\right\rfloor\right)_i \qquad \mathbb{K}^m\times\mathbb{K}^n\to\mathbb{K}^m
    $$
    """
    for i in range(len(v)):
        v[i] //= w[i]
    return v

def vecihadamardmod(v, w):
    r"""Return the elementwise remainder.
    
    $$
        \left(v_i \bmod w_i\right)_i \qquad \mathbb{K}^m\times\mathbb{K}^n\to\mathbb{K}^m
    $$
    """
    for i in range(len(v)):
        v[i] %= w[i]
    return v
