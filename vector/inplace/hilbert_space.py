from ..lazy import try_conjugate



__all__ = ('veciconj',)



def veciconj(v):
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
