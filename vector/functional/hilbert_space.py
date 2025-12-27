from operator import mul
from itertools import tee, zip_longest
from ..lazy import try_conjugate, veclconj
from operationcounter import sumprod_default



__all__ = ('vecconj', 'vecabs', 'vecabsq', 'vecdot', 'vecparallel')



def vecconj(v):
    r"""Return the complex conjugate.
    
    $$
        \vec{v}^* \qquad \mathbb{K}^n\to\mathbb{K}^n
    $$
    
    Trys to call a method `conjugate` on each element.
    If not found, simply keeps the element as is.
    
    Complexity
    ----------
    For a vector of length $n$ there will be
    
    - $n$ scalar conjugations (`conjugate`).
    """
    return tuple(veclconj(v))

def vecabs(v, weights=None, conjugate=False, zero=0):
    r"""Return the Euclidean/$\ell_{\mathbb{N}_0}^2$-norm.
    
    $$
        ||\vec{v}||_{\ell_{\mathbb{N}_0}^2}=\sqrt{\sum_iv_i^{(*)}v_i\omega_i} \qquad \mathbb{K}^n\to\mathbb{K}_0^+
    $$
    
    Returns the square root of [`vecabsq`][vector.functional.vecabsq].
    
    Complexity
    ----------
    For a vector of length $n$ there will be
    
    - $n$ scalar conjugations (`conjugate`) (if selected),
    - $n$/$2n$ scalar multiplications (`mul`) without/with weights,
    - $\begin{cases}n-1&n\ge1\\0&n\le1\end{cases}$ scalar additions (`add`) &
    - one `^0.5` call.
    
    See also
    --------
    - squared version without square root: [`vecabsq`][vector.functional.hilbert_space.vecabsq]
    """
    #hypot(*v) doesn't work for complex
    #math.sqrt doesn't work for complex and cmath.sqrt always returns complex
    return vecabsq(v, weights=weights, conjugate=conjugate, zero=zero)**0.5

def vecabsq(v, weights=None, conjugate=False, zero=0):
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
    vc, v = tee(v, 2)
    if conjugate:
        vc = veclconj(vc)
    
    if weights is None:
        return sumprod_default(vc, v, default=zero)
    else:
        return sumprod_default(map(mul, vc, v), weights, default=zero)

def vecdot(v, w, weights=None, conjugate=False, zero=0):
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
        v = veclconj(v)
    
    if weights is None:
        return sumprod_default(v, w, default=zero)
    else:
        return sumprod_default(map(mul, v, w), weights, default=zero)

def vecparallel(v, w, weights=None, conjugate=False, zero=0):
    r"""Return if two vectors are parallel.
    
    $$
        \vec{v}\parallel\vec{w} \qquad ||\vec{v}||\,||\vec{w}|| \overset{?}{=} |\vec{v}\vec{w}|^2 \qquad \mathbb{K}^m\times\mathbb{K}^n\to\mathbb{B}
    $$
    
    Complexity
    ----------
    Not yet perfect.
    """
    #doesn't work for exhaustible iterables
    #return vecabsq(v)*vecabsq(w) == abs(vecdot(v, w))**2
    v2, w2, vw = zero, zero, zero
    if weights is None:
        for vi, wi in zip_longest(v, w, fillvalue=zero):
            vic, wic = (try_conjugate(vi), try_conjugate(wi)) if conjugate else (vi, wi)
            v2 += vic * vi
            w2 += wic * wi
            vw += vic * wi
    else:
        for vi, wi, o in zip_longest(v, w, weights, fillvalue=zero):
            vic, wic = (try_conjugate(vi), try_conjugate(wi)) if conjugate else (vi, wi)
            v2 += vic * vi * o
            w2 += wic * wi * o
            vw += vic * wi * o
    return v2 * w2 == (try_conjugate(vw) if conjugate else vw) * vw
