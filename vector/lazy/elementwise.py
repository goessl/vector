from operator import truediv, floordiv, mod
from itertools import chain
from functools import partial
from operationcounter import MISSING, exception_generator, group_ordinal, prod_default



__all__ = ('veclhadamard', 'veclhadamardtruediv',
           'veclhadamardfloordiv', 'veclhadamardmod', 'veclhadamarddivmod',
           'veclhadamardmin', 'veclhadamardmax')



def veclhadamard(*vs):
    r"""Return the elementwise product of vectors.
    
    $$
        \left((\vec{v}_0)_i\cdot(\vec{v}_1)_i\cdot\cdots\right)_i \qquad \mathbb{K}^{n_0}\times\mathbb{K}^{n_1}\times\cdots\to\mathbb{K}^{\min_i n_i}
    $$
    """
    yield from map(partial(prod_default, default=MISSING), zip(*vs))

def veclhadamardtruediv(v, w):
    r"""Return the elementwise true division of two vectors.
    
    $$
        \left(\frac{v_i}{w_i}\right)_i \qquad \mathbb{K}^m\times\mathbb{K}^n\to\mathbb{K}^m
    $$
    """
    yield from map(truediv, v, chain(w, exception_generator(ZeroDivisionError)))

def veclhadamardfloordiv(v, w):
    r"""Return the elementwise floor division of two vectors.
    
    $$
        \left(\left\lfloor\frac{v_i}{w_i}\right\rfloor\right)_i \qquad \mathbb{K}^m\times\mathbb{K}^n\to\mathbb{K}^m
    $$
    """
    yield from map(floordiv, v, chain(w, exception_generator(ZeroDivisionError)))

def veclhadamardmod(v, w):
    r"""Return the elementwise mod of two vectors.
    
    $$
        \left(v_i \mod w_i\right)_i \qquad \mathbb{K}^m\times\mathbb{K}^n\to\mathbb{K}^m
    $$
    """
    yield from map(mod, v, chain(w, exception_generator(ZeroDivisionError)))

def veclhadamarddivmod(v, w):
    r"""Return the elementwise divmod of two vectors.
    
    $$
        \left(\left\lfloor\frac{v_i}{w_i}\right\rfloor\right)_i, \ \left(v_i \mod w_i\right)_i \qquad \mathbb{K}^n\times\mathbb{K}^m\to\mathbb{K}^n\times\mathbb{K}^n
    $$
    """
    yield from map(divmod, v, chain(w, exception_generator(ZeroDivisionError)))

def veclhadamardmin(*vs, key=None):
    r"""Return the elementwise minimum of vectors.
    
    $$
        \left(\min((\vec{v}_0)_i,(\vec{v}_1)_i,\cdots)\right)_i \qquad \mathbb{K}^{n_0}\times\mathbb{K}^{n_1}\times\cdots\to\mathbb{K}^{\max_i n_i}
    $$
    """
    yield from map(partial(min, key=key), group_ordinal(*vs))

def veclhadamardmax(*vs, key=None):
    r"""Return the elementwise maximum of vectors.
    
    $$
        \left(\max((\vec{v}_0)_i,(\vec{v}_1)_i,\cdots)\right)_i \qquad \mathbb{K}^{n_0}\times\mathbb{K}^{n_1}\times\cdots\to\mathbb{K}^{\max_i n_i}
    $$
    """
    yield from map(partial(max, key=key), group_ordinal(*vs))

def vechadamardminmax(*vs):
    pass
