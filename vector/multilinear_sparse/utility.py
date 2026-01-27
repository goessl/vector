from operator import add, sub
from itertools import starmap, zip_longest
from ..dense.elementwise import vechadamardmax



__all__ = ('tensrank', 'tensdim', 'tenseq',
           'tenstrim', 'tensitrim',
           'tensrshift', 'tenslshift')



def tensrank(t):
    r"""Return the rank.
    
    $$
        \text{rank}\,t
    $$
    """
    return max(map(len, t.keys()), default=0)

def tensdim(t):
    r"""Return the dimensionalities.
    
    $$
        \dim t
    $$
    """
    return tuple(si+1 for si in vechadamardmax(*t.keys()))

def tenseq(s, t):
    r"""Return whether two tensors are equal.
    
    $$
        s\overset{?}{=}t
    $$
    """
    for i in s.keys()&t.keys():
        if i not in t:
            if bool(s[i]):
                return False
        elif i not in s:
            if bool(t[i]):
                return False
        else:
            if s[i] != t[i]:
                return False
    return True

def tenstrim(t, tol=None):
    """Remove all near zero (`abs(t_i)<=tol`) coefficients.
    
    `tol` may also be `None`,
    then all coefficients that evaluate to `False` are trimmed.
    
    Notes
    -----
    - Cutting of elements that are `abs(t_i)<=tol` instead of `abs(t_i)<tol` to
    allow cutting of elements that are exactly zero by `trim(t, 0)` instead
    of `trim(t, sys.float_info.min)`.
    """
    if tol is None:
        return {i:ti for i, ti in t.items() if ti}
    else:
        return {i:ti for i, ti in t.items() if abs(ti)>tol}

def tensitrim(t, tol=None):
    """Remove all near zero (`abs(t_i)<=tol`) coefficients.
    
    `tol` may also be `None`,
    then all coefficients that evaluate to `False` are trimmed.
    
    Notes
    -----
    - Cutting of elements that are `abs(t_i)<=tol` instead of `abs(t_i)<tol` to
    allow cutting of elements that are exactly zero by `trim(t, 0)` instead
    of `trim(t, sys.float_info.min)`.
    """
    if tol is None:
        indices_to_del = tuple(i for i, ti in t.items() if not ti)
    else:
        indices_to_del = tuple(i for i, ti in t.items() if abs(ti)<=tol)
    
    for i in indices_to_del:
        del t[i]
    return t

def tensrshift(t, n):
    """Shift coefficients up."""
    #raw vector addition of indices prolly faster than vecadd
    return {tuple(starmap(add, zip_longest(i, n, fillvalue=0))):ti
            for i, ti in t.items()}

def tenslshift(t, n):
    """Shift coefficients down."""
    r = {}
    for i, ti in t.items():
        i = tuple(starmap(sub, zip_longest(i, n, fillvalue=0)))
        if all(ii>=0 for ii in i):
            r[i] = ti
    return r
