from ..lazy import veclhadamard, veclhadamardtruediv, veclhadamardfloordiv, veclhadamardmod, veclhadamarddivmod, veclhadamardmin, veclhadamardmax
from itertools import chain, islice, repeat
from typing import Any
from collections.abc import Iterable, Sequence, MutableSequence, Callable



__all__ = ('vechadamard',         'vecihadamard',
           'vechadamardtruediv',  'vecihadamardtruediv',
           'vechadamardfloordiv', 'vecihadamardfloordiv',
           'vechadamardmod',      'vecihadamardmod',
           'vechadamarddivmod',
           'vechadamardmin', 'vechadamardmax')



def vechadamard(*vs:Iterable[Any]) -> tuple[Any,...]:
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

def vecihadamard(v:MutableSequence[Any], *ws:Sequence[Any]) -> MutableSequence[Any]:
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


def vechadamardtruediv(v:Iterable[Any], w:Iterable[Any]) -> tuple[Any,...]:
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

def vecihadamardtruediv(v:MutableSequence[Any], w:Iterable[Any]) -> MutableSequence[Any]:
    r"""Return the elementwise true quotient.
    
    $$
        \left(\frac{v_i}{w_i}\right)_i \qquad \mathbb{K}^m\times\mathbb{K}^n\to\mathbb{K}^m
    $$
    """
    for i, wi in enumerate(islice(chain(w, repeat(0)), len(v))):
        v[i] /= wi
    return v


def vechadamardfloordiv(v:Iterable[Any], w:Iterable[Any]) -> tuple[Any,...]:
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

def vecihadamardfloordiv(v:MutableSequence[Any], w:Iterable[Any]) -> MutableSequence[Any]:
    r"""Return the elementwise floor quotient.
    
    $$
        \left(\left\lfloor\frac{v_i}{w_i}\right\rfloor\right)_i \qquad \mathbb{K}^m\times\mathbb{K}^n\to\mathbb{K}^m
    $$
    """
    for i, wi in enumerate(islice(chain(w, repeat(0)), len(v))):
        v[i] //= wi
    return v


def vechadamardmod(v:Iterable[Any], w:Iterable[Any]) -> tuple[Any,...]:
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

def vecihadamardmod(v:MutableSequence[Any], w:Iterable[Any]) -> MutableSequence[Any]:
    r"""Return the elementwise remainder.
    
    $$
        \left(v_i \bmod w_i\right)_i \qquad \mathbb{K}^m\times\mathbb{K}^n\to\mathbb{K}^m
    $$
    """
    for i, wi in enumerate(islice(chain(w, repeat(0)), len(v))):
        v[i] %= wi
    return v


def vechadamarddivmod(v:Iterable[Any], w:Iterable[Any]) -> tuple[Any,...]:
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


def vechadamardmin(*vs:Iterable[Any], key:Callable[[Any], Any]|None=None) -> tuple[Any,...]:
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

def vechadamardmax(*vs:Iterable[Any], key:Callable[[Any], Any]|None=None) -> tuple[Any,...]:
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
