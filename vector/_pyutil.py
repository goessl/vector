__all__ = ('try_conjugate', )



def try_conjugate(x):
    r"""Return the complex conjugate.
    
    $$
        x^* \qquad \mathbb{K}\to\mathbb{K}
    $$
    
    Tries to call a method `conjugate`.
    If not found, simply returns the element as is.
    
    Python implementation.
    """
    conj = getattr(x, 'conjugate', None)
    return conj() if callable(conj) else x
