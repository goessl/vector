from operator import mul
from itertools import tee
from ..util import try_conjugate
from iteration import sumprod_default
from typing import Any, TypeVar
from collections.abc import Callable, Generator, Iterable, Iterator, MutableSequence, Sequence



__all__ = ('vecconj', 'veciconj',
           'vecabs',
           'vecabsq', 'vecabsqs',
           'vecdot',  'vecdots')



V = TypeVar('V')
S = TypeVar('S', bound=Sequence)
M = TypeVar('M', bound=MutableSequence)



def vecconj(v:Iterable, factory:Callable[[Iterable],V]|None=None) -> V:
    r"""Return the complex conjugate.
    
    $$
        \vec{v}^* \qquad \mathbb{K}^n\to\mathbb{K}^n
    $$
    
    Tries to call a method `conjugate` on each element.
    If not found, simply keeps the element as is.
    
    Complexity
    ----------
    For a vector of length $n$ there will be
    
    - $n$ scalar conjugations (`conjugate`).
    """
    factory = factory or (iter if isinstance(v, Iterator) else type(v))
    return factory(map(try_conjugate, v))

def veciconj(v:M) -> M:
    r"""Complex conjugate.
    
    $$
        \vec{v} = \vec{v}^* \qquad \mathbb{K}^n\to\mathbb{K}^n
    $$
    
    Tries to call a method `conjugate` on each element.
    If not found, simply keeps the element as is.
    """
    for i, vi in enumerate(v):
        v[i] = try_conjugate(vi)
    return v


def vecabs(v:Iterable, weights:Iterable|None=None, conjugate:bool=False, zero:Any=0) -> Any:
    r"""Return the Euclidean/$\ell_{\mathbb{N}_0}^2$-norm.
    
    $$
        ||\vec{v}||_{\ell_{\mathbb{N}_0}^2}=\sqrt{\sum_iv_i^{(*)}v_i\omega_i} \qquad \mathbb{K}^n\to\mathbb{K}_0^+
    $$
    
    Returns the square root of [`vecabsq`][vector.dense.vecabsq].
    
    Complexity
    ----------
    For a vector of length $n$ there will be
    
    - $n$ scalar conjugations (`conjugate`) (if selected),
    - $n$/$2n$ scalar multiplications (`mul`) without/with weights,
    - $\begin{cases}n-1&n\ge1\\0&n\le1\end{cases}$ scalar additions (`add`) &
    - one `^0.5` call.
    
    See also
    --------
    - squared version without square root: [`vecabsq`][vector.dense.hilbertspace.vecabsq]
    """
    #hypot(*v) doesn't work for complex
    #math.sqrt doesn't work for complex and cmath.sqrt always returns complex
    return vecabsq(v, weights=weights, conjugate=conjugate, zero=zero)**0.5


def vecabsqs(v:Iterable, weights:Iterable|None=None, conjugate:bool=False) -> Generator:
    r"""Return the absolute squares.
    
    $$
        v_i^{(*)}v_i\omega_i
    $$
    
    Complexity
    ----------
    For a vector of length $n$ there will be
    
    - $n$ scalar conjugations (`conjugate`) (if selected),
    - $\begin{cases}n-1&n\ge1\\0&n\le1\end{cases}$ scalar additions (`add`) &
    - $n$/$2n$ scalar multiplications (`mul`) without/with weights.
    
    Notes
    -----
    Reasons why it exists:
    
    - Occurs in math.
    - Most importantly: type independent because it doesn't use `sqrt`.
    
    References
    ----------
    - <https://docs.python.org/3/library/itertools.html#itertools-recipes>: `sum_of_squares`
    """
    yield from vecdots(*tee(v, 2), weights, conjugate)

def vecabsq(v:Iterable, weights:Iterable|None=None, conjugate:bool=False, zero:Any=0) -> Any:
    r"""Return the sum of absolute squares.
    
    $$
        ||\vec{v}||_{\ell_{\mathbb{N}_0}^2}^2=\sum_iv_i^{(*)}v_i\omega_i \qquad \mathbb{K}^n\to\mathbb{K}_0^+
    $$
    
    Complexity
    ----------
    For a vector of length $n$ there will be
    
    - $n$ scalar conjugations (`conjugate`) (if selected),
    - $\begin{cases}n-1&n\ge1\\0&n\le1\end{cases}$ scalar additions (`add`) &
    - $n$/$2n$ scalar multiplications (`mul`) without/with weights.
    
    Notes
    -----
    Reasons why it exists:
    
    - Occurs in math.
    - Most importantly: type independent because it doesn't use `sqrt`.
    
    References
    ----------
    - <https://docs.python.org/3/library/itertools.html#itertools-recipes>: `sum_of_squares`
    """
    return vecdot(*tee(v, 2), weights, conjugate, zero)


def vecdots(v:Iterable, w:Iterable, weights:Iterable|None=None, conjugate:bool=False) -> Generator:
    r"""Return the elementwise product.
    
    $$
        v_i^{(*)}w_i\omega_i
    $$
    
    Complexity
    ----------
    For two vectors of lengths $n$ & $m$ there will be
    
    - $\min\{n, m\}$ scalar conjugations (`conjugate`) (if selected),
    - $\min\{n, m\}$/$2\min\{n, m\}$ scalar multiplications (`mul` without/with weights) &
    - $\begin{cases}\min\{n, m\}-1&n\ge1\land m\ge1\\0&n\le1\lor m\le1\end{cases}$ scalar additions (`add`).
    """
    if conjugate:
        v = vecconj(v, iter)
    
    if weights is None:
        yield from map(mul, v, w)
    else:
        yield from map(mul, map(mul, v, w), weights)


def vecdot(v:Iterable, w:Iterable, weights:Iterable|None=None, conjugate:bool=False, zero:Any=0) -> Any:
    r"""Return the inner product.
    
    $$
        \left<\vec{v}\mid\vec{w}\right>_{\ell_{\mathbb{N}_0}^2}=\sum_iv_i^{(*)}w_i\omega_i \qquad \mathbb{K}^m\times\mathbb{K}^n\to\mathbb{K}
    $$
    
    Complexity
    ----------
    For two vectors of lengths $n$ & $m$ there will be
    
    - $\min\{n, m\}$ scalar conjugations (`conjugate`) (if selected),
    - $\min\{n, m\}$/$2\min\{n, m\}$ scalar multiplications (`mul` without/with weights) &
    - $\begin{cases}\min\{n, m\}-1&n\ge1\land m\ge1\\0&n\le1\lor m\le1\end{cases}$ scalar additions (`add`).
    """
    if conjugate:
        v = vecconj(v, iter)
    #don't sum(vecldot) but rather use sumprod explicitly for improved float precision
    if weights is None:
        return sumprod_default(v, w, default=zero)
    else:
        return sumprod_default(map(mul, v, w), weights, default=zero)
