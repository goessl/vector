from operator import mul
from itertools import tee
from ..util import try_conjugate
from collections.abc import Generator, Iterable



__all__ = ('veclconj', 'veclabsq', 'vecldot')



def veclconj(v:Iterable) -> Generator:
    r"""Return the complex conjugate.
    
    $$
        \vec{v}^* \qquad \mathbb{K}^n\to\mathbb{K}^n
    $$
    
    Tries to call a method `conjugate` on each element.
    If not found, simply keeps the element as is.
    """
    yield from map(try_conjugate, v)


def veclabsq(v:Iterable, weights:Iterable|None=None, conjugate:bool=False) -> Generator:
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
    yield from vecldot(*tee(v, 2), weights, conjugate)

def vecldot(v:Iterable, w:Iterable, weights:Iterable|None=None, conjugate:bool=False) -> Generator:
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
        v = veclconj(v)
    
    if weights is None:
        yield from map(mul, v, w)
    else:
        yield from map(mul, map(mul, v, w), weights)
