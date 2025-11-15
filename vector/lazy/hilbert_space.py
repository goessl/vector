__all__ = ('try_conjugate', 'veclconj')



def try_conjugate(x):
    r"""Return the complex conjugate of a scalar.
    
    $$
        x^* \qquad \mathbb{K}\to\mathbb{K}
    $$
    
    Trys to call a method `conjugate`.
    If not found, simply returns the element as is.
    """
    #try:
    #    return x.conjugate()
    #except AttributeError:
    #    return x
    #could throw an AttibuteError from somewhere deeper
    conj = getattr(x, 'conjugate', None)
    return conj() if callable(conj) else x

def veclconj(v):
    r"""Return the elementwise complex conjugate of a vector.
    
    $$
        \vec{v}^* \qquad \mathbb{K}^n\to\mathbb{K}^n
    $$
    
    Trys to call a method `conjugate` on each element.
    If not found, simply keeps the element as is.
    """
    yield from map(try_conjugate, v)
