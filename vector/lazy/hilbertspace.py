from ..util import try_conjugate



__all__ = ('veclconj', )



def veclconj(v):
    r"""Return the complex conjugate.
    
    $$
        \vec{v}^* \qquad \mathbb{K}^n\to\mathbb{K}^n
    $$
    
    Tries to call a method `conjugate` on each element.
    If not found, simply keeps the element as is.
    """
    yield from map(try_conjugate, v)
