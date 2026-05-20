from ..lazy import veclhadamard, veclhadamardtruediv, veclhadamardfloordiv, veclhadamardmod, veclhadamarddivmod, veclhadamardmin, veclhadamardmax
from itertools import chain, islice, repeat
from typing import Any, TypeVar
from collections.abc import Iterable, Sequence, MutableSequence, Callable



__all__ = ('vechadamard',         'vecihadamard',
           'vechadamardtruediv',  'vecihadamardtruediv',
           'vechadamardfloordiv', 'vecihadamardfloordiv',
           'vechadamardmod',      'vecihadamardmod',
           'vechadamarddivmod',
           'vechadamardmin', 'vechadamardmax')



S = TypeVar('S', bound=Sequence)
M = TypeVar('M', bound=MutableSequence)



def vechadamard(*vs:Iterable, factory:Callable[[Iterable],S]|None=None) -> S:
    r"""Return the elementwise product.
    
    $$
        \left((\vec{v}_0)_i\cdot(\vec{v}_1)_i\cdot\cdots\right)_i \qquad \mathbb{K}^{n_0}\times\mathbb{K}^{n_1}\times\cdots\to\mathbb{K}^{\min_i n_i}
    $$
    
    Complexity
    ----------
    For vectors of lengths $n_1, n_2, \dots, n_N$ there will be
    
    - $\begin{cases}(N-1)\min_in_i&N\ge1\land\min_in_i\ge1\\0&N\le1\lor\min_in_i=0\end{cases}$ scalar multiplications (`mul`).
    """
    if factory is None:
        factory = type(vs[0]) if vs else tuple
    return factory(veclhadamard(*vs))

def vecihadamard(v:M, *ws:Sequence) -> M:
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


def vechadamardtruediv(v:Iterable, w:Iterable, factory:Callable[[Iterable],S]|None=None) -> S:
    r"""Return the elementwise true quotient.
    
    $$
        \left(\frac{v_i}{w_i}\right)_i \qquad \mathbb{K}^m\times\mathbb{K}^n\to\mathbb{K}^m
    $$
    
    Complexity
    ----------
    For two vectors of lengths $n$ & $m$ there will be
    
    - $n$ scalar true divisions (`truediv`).
    """
    factory = type(v) if factory is None else factory
    return factory(veclhadamardtruediv(v, w))

def vecihadamardtruediv(v:M, w:Iterable) -> M:
    r"""Return the elementwise true quotient.
    
    $$
        \left(\frac{v_i}{w_i}\right)_i \qquad \mathbb{K}^m\times\mathbb{K}^n\to\mathbb{K}^m
    $$
    """
    for i, wi in enumerate(islice(chain(w, repeat(0)), len(v))):
        v[i] /= wi
    return v


def vechadamardfloordiv(v:Iterable, w:Iterable, factory:Callable[[Iterable],S]|None=None) -> S:
    r"""Return the elementwise floor quotient.
    
    $$
        \left(\left\lfloor\frac{v_i}{w_i}\right\rfloor\right)_i \qquad \mathbb{K}^m\times\mathbb{K}^n\to\mathbb{K}^m
    $$
    
    Complexity
    ----------
    For two vectors of lengths $n$ & $m$ there will be
    
    - $n$ scalar floor divisions (`floordiv`).
    """
    factory = type(v) if factory is None else factory
    return factory(veclhadamardfloordiv(v, w))

def vecihadamardfloordiv(v:M, w:Iterable) -> M:
    r"""Return the elementwise floor quotient.
    
    $$
        \left(\left\lfloor\frac{v_i}{w_i}\right\rfloor\right)_i \qquad \mathbb{K}^m\times\mathbb{K}^n\to\mathbb{K}^m
    $$
    """
    for i, wi in enumerate(islice(chain(w, repeat(0)), len(v))):
        v[i] //= wi
    return v


def vechadamardmod(v:Iterable, w:Iterable, factory:Callable[[Iterable],S]|None=None) -> S:
    r"""Return the elementwise remainder.
    
    $$
        \left(v_i \bmod w_i\right)_i \qquad \mathbb{K}^m\times\mathbb{K}^n\to\mathbb{K}^m
    $$
    
    Complexity
    ----------
    For two vectors of lengths $n$ & $m$ there will be
    
    - $n$ scalar modulos (`mod`).
    """
    factory = type(v) if factory is None else factory
    return factory(veclhadamardmod(v, w))

def vecihadamardmod(v:M, w:Iterable) -> M:
    r"""Return the elementwise remainder.
    
    $$
        \left(v_i \bmod w_i\right)_i \qquad \mathbb{K}^m\times\mathbb{K}^n\to\mathbb{K}^m
    $$
    """
    for i, wi in enumerate(islice(chain(w, repeat(0)), len(v))):
        v[i] %= wi
    return v


def vechadamarddivmod(v:Iterable, w:Iterable, factory:Callable[[Iterable],S]|None=None) -> tuple[S, S]:
    r"""Return the elementwise floor quotient and remainder.
    
    $$
        \left(\left\lfloor\frac{v_i}{w_i}\right\rfloor\right)_i, \ \left(v_i \bmod w_i\right)_i \qquad \mathbb{K}^n\times\mathbb{K}^m\to\mathbb{K}^n\times\mathbb{K}^n
    $$
    
    Complexity
    ----------
    For two vectors of lengths $n$ & $m$ there will be
    
    - $n$ scalar divmods (`divmod`).
    """
    factory = type(v) if factory is None else factory
    q, r = [], []
    for qi, ri in veclhadamarddivmod(v, w):
        q.append(qi)
        r.append(ri)
    return factory(q), factory(r)


def vechadamardmin(*vs:Iterable, key:Callable[[Any], Any]|None=None, factory:Callable[[Iterable],S]|None=None) -> S:
    r"""Return the elementwise minimum.
    
    $$
        \left(\min((\vec{v}_0)_i,(\vec{v}_1)_i,\cdots)\right)_i \qquad \mathbb{K}^{n_0}\times\mathbb{K}^{n_1}\times\cdots\to\mathbb{K}^{\max_i n_i}
    $$
    
    Complexity
    ----------
    For two vectors of lengths $n$ & $m$ there will be
    
    - $\min\{n, m\}$ comparisons (`lt`).
    """
    if factory is None:
        factory = type(vs[0]) if vs else tuple
    return factory(veclhadamardmin(*vs, key=key))

def vechadamardmax(*vs:Iterable, key:Callable[[Any], Any]|None=None, factory:Callable[[Iterable],S]|None=None) -> S:
    r"""Return the elementwise maximum.
    
    $$
        \left(\max((\vec{v}_0)_i,(\vec{v}_1)_i,\cdots)\right)_i \qquad \mathbb{K}^{n_0}\times\mathbb{K}^{n_1}\times\cdots\to\mathbb{K}^{\max_i n_i}
    $$
    
    Complexity
    ----------
    For two vectors of lengths $n$ & $m$ there will be
    
    - $\min\{n, m\}$ comparisons (`gt`).
    """
    if factory is None:
        factory = type(vs[0]) if vs else tuple
    return factory(veclhadamardmax(*vs, key=key))
