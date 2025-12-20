from ..functional import vectrim, vecadd, vecsub, vechadamardmax



__all__ = ('tensrank', 'tensdim', 'tenstrim', 'tensround')



def tensrank(t):
    r"""Return the rank of a tensor.
    
    $$
        \text{rank}\,t
    $$
    """
    return max(map(len, t.keys()), default=0)

def tensdim(t):
    r"""Return the dimensionalities of a tensor.
    
    $$
        \dim t
    $$
    """
    return tuple(si+1 for si in vechadamardmax(*t.keys()))

def tenseq(s, t):
    """Return if two tensors are equal."""
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

def tenstrim(t, tol=1e-9):
    """Remove all near zero (`abs(t_i)<=tol`) coefficients."""
    if tol is None:
        return {i:ti for i, ti in t.items() if ti}
    else:
        return {i:ti for i, ti in t.items() if abs(ti)>tol}

def tensround(t, ndigits=0):
    r"""Round all coefficients to the given precision.
    
    $$
        (\text{round}_\text{ndigits}(v_i))_i
    $$
    """
    return {i:round(ti, ndigits) for i, ti in t.items()}

def tensrshift(t, n):
    """Pad `n` many zeros to the beginning of the tensor."""
    return {vecadd(i, n):ti for i, ti in t.items()}

def tenslshift(t, n):
    """Remove `n` many coefficients at the beginning of the tensor."""
    return {vectrim(vecsub(i, n)):ti for i, ti in t.items() if all(ii>=0 for ii in vecsub(i, n))}
