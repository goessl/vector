from operator import truediv, floordiv, mod
from itertools import chain
from functools import partial
from operationcounter import MISSING, raiser, group_ordinal, prod_default
from typing import Any
from collections.abc import Callable, Generator, Iterable



__all__ = ('veclhadamard', 'veclhadamardtruediv',
           'veclhadamardfloordiv', 'veclhadamardmod', 'veclhadamarddivmod',
           'veclhadamardmin', 'veclhadamardmax')



def veclhadamard(*vs:Iterable) -> Generator:
    r"""Return the elementwise product.
    
    $$
        \left((\vec{v}_0)_i\cdot(\vec{v}_1)_i\cdot\cdots\right)_i \qquad \mathbb{K}^{n_0}\times\mathbb{K}^{n_1}\times\cdots\to\mathbb{K}^{\min_i n_i}
    $$
    """
    yield from map(partial(prod_default, default=MISSING), zip(*vs))

def veclhadamardtruediv(v:Iterable, w:Iterable) -> Generator:
    r"""Return the elementwise true quotient.
    
    $$
        \left(\frac{v_i}{w_i}\right)_i \qquad \mathbb{K}^m\times\mathbb{K}^n\to\mathbb{K}^m
    $$
    """
    yield from map(truediv, v, chain(w, raiser(ZeroDivisionError)))

def veclhadamardfloordiv(v:Iterable, w:Iterable) -> Generator:
    r"""Return the elementwise floor quotient.
    
    $$
        \left(\left\lfloor\frac{v_i}{w_i}\right\rfloor\right)_i \qquad \mathbb{K}^m\times\mathbb{K}^n\to\mathbb{K}^m
    $$
    """
    yield from map(floordiv, v, chain(w, raiser(ZeroDivisionError)))

def veclhadamardmod(v:Iterable, w:Iterable) -> Generator:
    r"""Return the elementwise remainder.
    
    $$
        \left(v_i \bmod w_i\right)_i \qquad \mathbb{K}^m\times\mathbb{K}^n\to\mathbb{K}^m
    $$
    """
    yield from map(mod, v, chain(w, raiser(ZeroDivisionError)))

def veclhadamarddivmod(v:Iterable, w:Iterable) -> Generator[tuple[Any,Any]]:
    r"""Return the elementwise floor quotient and remainder.
    
    $$
        \left(\left\lfloor\frac{v_i}{w_i}\right\rfloor\right)_i, \ \left(v_i \bmod w_i\right)_i \qquad \mathbb{K}^n\times\mathbb{K}^m\to\mathbb{K}^n\times\mathbb{K}^n
    $$
    """
    yield from map(divmod, v, chain(w, raiser(ZeroDivisionError)))

def veclhadamardmin(*vs:Iterable, key:Callable[[Any],Any]|None=None) -> Generator:
    r"""Return the elementwise minimum.
    
    $$
        \left(\min((\vec{v}_0)_i,(\vec{v}_1)_i,\cdots)\right)_i \qquad \mathbb{K}^{n_0}\times\mathbb{K}^{n_1}\times\cdots\to\mathbb{K}^{\max_i n_i}
    $$
    """
    yield from map(partial(min, key=key), group_ordinal(*vs))

def veclhadamardmax(*vs:Iterable, key:Callable[[Any],Any]|None=None) -> Generator:
    r"""Return the elementwise maximum.
    
    $$
        \left(\max((\vec{v}_0)_i,(\vec{v}_1)_i,\cdots)\right)_i \qquad \mathbb{K}^{n_0}\times\mathbb{K}^{n_1}\times\cdots\to\mathbb{K}^{\max_i n_i}
    $$
    """
    yield from map(partial(max, key=key), group_ordinal(*vs))
