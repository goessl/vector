from ..lazy import try_conjugate
from operationcounter import sum_default, sumprod_default



__all__ = ('vecsconj', 'vecsabs', 'vecsabsq', 'vecsdot', 'vecsparallel')



def vecsconj(v):
    r"""Return the complex conjugate.
    
    $$
        \vec{v}^*
    $$
    
    Trys to call a method `conjugate` on each element.
    If not found, simply keeps the element as is.
    
    Complexity
    ----------
    For a vector of $n$ elements there will be
    
    - $n$ scalar conjugations.
    """
    return {i:try_conjugate(vi) for i, vi in v.items()}

def vecsabs(v, weights=None, conjugate=False, zero=0):
    r"""Return the Euclidean/$\ell_{\mathbb{N}_0}^2$-norm.
    
    $$
        ||\vec{v}||_{\ell_{\mathbb{N}_0}^2}=\sqrt{\sum_iv_i^{(*)}v_i\omega_i}
    $$
    
    Returns the square root of [`vecsabsq`][vector.sparse.hilbert_space.vecsabsq].
    
    Complexity
    ----------
    For a vector of $n$ elements there will be
    
    - $n$ scalar conjugations (`conjugate`) (if selected),
    - $n$/$2n$ scalar multiplications (`mul`) without/with weights,
    - $\begin{cases}n-1&n\ge1\\0&n\le1\end{cases}$ scalar additions (`add`) &
    - one `^0.5` call.
    
    See also
    --------
    - squared version without square root: [`vecsabsq`][vector.sparse.hilbert_space.vecsabsq]
    """
    return vecsabsq(v, weights=weights, conjugate=conjugate, zero=zero)**0.5

def vecsabsq(v, weights=None, conjugate=False, zero=0):
    r"""Return the sum of absolute squares.
    
    $$
        ||\vec{v}||_{\ell_{\mathbb{N}_0}^2}^2=\sum_iv_i^{(*)}v_i\omega_i \qquad \mathbb{K}^n\to\mathbb{K}_0^+
    $$
    
    Notes
    -----
    Reasons why it exists:
    
    - Occurs in math.
    - Most importantly: type independent because it doesn't use `sqrt`.
    
    References
    ----------
    - <https://docs.python.org/3/library/itertools.html#itertools-recipes>: `sum_of_squares`
    """
    if weights is None:
        if not conjugate:
            return sumprod_default(v.values(), v.values(), default=zero)
        else:
            return sumprod_default(map(try_conjugate, v.values()), v.values(), default=zero)
    else:
        if not conjugate:
            return sum_default((vi*vi*weights[i] for i, vi in v.items()), default=zero)
        else:
            return sum_default((try_conjugate(vi)*vi*weights[i] for i, vi in v.items()), default=zero)

def vecsdot(v, w, weights=None, conjugate=False, zero=0):
    r"""Return the inner product.
    
    $$
        \left<\vec{v}\mid\vec{w}\right>_{\ell_{\mathbb{N}_0}^2}=\sum_iv_i^{(*)}w_i\omega_i
    $$
    """
    if weights is None:
        if not conjugate:
            return sum_default((v[k]*w[k] for k in v.keys()&w.keys()), default=zero)
        else:
            return sum_default((try_conjugate(v[k])*w[k] for k in v.keys()&w.keys()), default=zero)
    else:
        if not conjugate:
            return sum_default((v[k]*w[k]*weights[k] for k in v.keys()&w.keys()), default=zero)
        else:
            return sum_default((try_conjugate(v[k])*w[k]*weights[k] for k in v.keys()&w.keys()), default=zero)

def vecsparallel(v, w, weights=None, conjugate=False, zero=0):
    r"""Return if two vectors are parallel.
    
    $$
        \vec{v}\parallel\vec{w} \qquad ||\vec{v}||\,||\vec{w}|| \overset{?}{=} |\vec{v}\vec{w}|^2
    $$
    """
    vv = vecsabsq(v, weights=weights, conjugate=conjugate, zero=zero)
    ww = vecsabsq(w, weights=weights, conjugate=conjugate, zero=zero)
    vw = vecsdot(v, w, weights=weights, conjugate=conjugate, zero=zero)
    return vv * ww == (try_conjugate(vw) if conjugate else vw) * vw
