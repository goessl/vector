from typing import Any
from collections.abc import Mapping, MutableMapping



__all__ = ('tenspos',             'tensipos',
           'tensneg',             'tensineg',
           'tensadd',             'tensiadd',
           'tensaddc',            'tensiaddc',
           'tenssub',             'tensisub',
           'tenssubc',            'tensisubc',
           'tensmul', 'tensrmul', 'tensimul',
           'tenstruediv',         'tensitruediv',
           'tensfloordiv',        'tensifloordiv',
           'tensmod',             'tensimod',
           'tensdivmod')



def tenspos(t:Mapping[tuple[int,...],Any]) -> dict[tuple[int,...],Any]:
    """Return the identity.
    
    $$
        +t
    $$
    """
    return {i:+ti for i, ti in t.items()}

def tensipos(t:MutableMapping[tuple[int,...],Any]) -> MutableMapping[tuple[int,...],Any]:
    """Apply unary plus.
    
    $$
        t = +t
    $$
    """
    for i, ti in t.items():
        t[i] = +ti
    return t

def tensneg(t:Mapping[tuple[int,...],Any]) -> dict[tuple[int,...],Any]:
    """Return the negation.
    
    $$
        -t
    $$
    """
    return {i:-ti for i, ti in t.items()}

def tensineg(t:dict[tuple[int,...],Any]) -> dict[tuple[int,...],Any]:
    """Negate.
    
    $$
        t = -t
    $$
    """
    for i, ti in t.items():
        t[i] = -ti
    return t

def tensadd(*ts:Mapping[tuple[int,...],Any]) -> dict[tuple[int,...],Any]:
    r"""Return the sum.
    
    $$
        t_0 + t_1 + \cdots
    $$
    
    See also
    --------
    - for sum on a single coefficient: [`tensaddc`][vector.multilinear_sparse.vectorspace.tensaddc]
    """
    r:dict[tuple[int,...],Any] = dict(ts[0]) if ts else {}
    for t in ts[1:]:
        for i, ti in t.items():
            if i in r:
                r[i] += ti
            else:
                r[i] = +ti
    return r

def tensiadd(s:MutableMapping[tuple[int,...],Any], *ts:Mapping[tuple[int,...],Any]) -> MutableMapping[tuple[int,...],Any]:
    r"""Add.
    
    $$
        s += t_0 + t_1 + \cdots
    $$
    
    See also
    --------
    - for sum on a single coefficient: [`tensiaddc`][vector.multilinear_sparse.vectorspace.tensiaddc]
    """
    for t in ts:
        for i, ti in t.items():
            if i in s:
                s[i] += ti
            else:
                s[i] = +ti
    return s

def tensaddc(t:Mapping[tuple[int,...],Any], c:Any, i:tuple[int,...]=()) -> dict[tuple[int,...],Any]:
    """Return the sum with a basis tensor.
    
    $$
        t + ce_i
    $$
    
    See also
    --------
    - for sum on more coefficients: [`tensadd`][vector.multilinear_sparse.vectorspace.tensadd]
    """
    r:dict[tuple[int,...],Any] = dict(t)
    if i in r:
        r[i] += c
    else:
        r[i] = +c
    return r

def tensiaddc(t:MutableMapping[tuple[int,...],Any], c:Any, i:tuple[int,...]=()) -> MutableMapping[tuple[int,...],Any]:
    """Add a basis tensor.
    
    $$
        t += ce_i
    $$
    
    See also
    --------
    - for sum on more coefficients: [`tensiadd`][vector.multilinear_sparse.vectorspace.tensiadd]
    """
    if i in t:
        t[i] += c
    else:
        t[i] = +c
    return t

def tenssub(s:Mapping[tuple[int,...],Any], t:Mapping[tuple[int,...],Any]) -> dict[tuple[int,...],Any]:
    """Return the difference.
    
    $$
        s - t
    $$
    
    See also
    --------
    - for difference on a single coefficient: [`tenssubc`][vector.multilinear_sparse.vectorspace.tenssubc]
    """
    r:dict[tuple[int,...],Any] = dict(s)
    for i, ti in t.items():
        if i in r:
            r[i] -= ti
        else:
            r[i] = -ti
    return r

def tensisub(s:MutableMapping[tuple[int,...],Any], t:Mapping[tuple[int,...],Any]) -> MutableMapping[tuple[int,...],Any]:
    """Subtract.
    
    $$
        s -= t
    $$
    
    See also
    --------
    - for difference on a single coefficient: [`tensisubc`][vector.multilinear_sparse.vectorspace.tensisubc]
    """
    for i, ti in t.items():
        if i in s:
            s[i] -= ti
        else:
            s[i] = -ti
    return s

def tenssubc(t:Mapping[tuple[int,...],Any], c:Any, i:tuple[int,...]=()) -> dict[tuple[int,...],Any]:
    """Return the difference with a basis tensor.
    
    $$
        t - ce_i
    $$
    
    See also
    --------
    - for difference on more coefficients: [`tenssub`][vector.multilinear_sparse.vectorspace.tenssub]
    """
    r:dict[tuple[int,...],Any] = dict(t)
    if i in r:
        r[i] -= c
    else:
        r[i] = -c
    return r

def tensisubc(t:MutableMapping[tuple[int,...],Any], c:Any, i:tuple[int,...]=()) -> MutableMapping[tuple[int,...],Any]:
    """Subtract a basis tensor.
    
    $$
        t -= ce_i
    $$
    
    See also
    --------
    - for difference on more coefficients: [`tensisub`][vector.multilinear_sparse.vectorspace.tensisub]
    """
    if i in t:
        t[i] -= c
    else:
        t[i] = -c
    return t

def tensmul(t:Mapping[tuple[int,...],Any], a:Any) -> dict[tuple[int,...],Any]:
    """Return the product.
    
    $$
        ta
    $$
    """
    return {i:ti*a for i, ti in t.items()}

def tensrmul(a:Any, t:Mapping[tuple[int,...],Any]) -> dict[tuple[int,...],Any]:
    """Return the product.
    
    $$
        at
    $$
    """
    return {i:a*ti for i, ti in t.items()}

def tensimul(t:MutableMapping[tuple[int,...],Any], a:Any) -> MutableMapping[tuple[int,...],Any]:
    r"""Multiply.
    
    $$
        t \cdot= a
    $$
    """
    for i in t:
        t[i] *= a
    return t

def tenstruediv(t:Mapping[tuple[int,...],Any], a:Any) -> dict[tuple[int,...],Any]:
    r"""Return the true quotient.
    
    $$
        \frac{t}{a}
    $$
    
    Notes
    -----
    Why called `truediv` instead of `div`?
    
    - `div` would be more appropriate for an absolutely clean mathematical
    implementation, that doesn't care about the language used. But the package
    might be used for pure integers/integer arithmetic, so both, `truediv`
    and `floordiv` operations have to be provided, and none should be
    privileged over the other by getting the universal `div` name.
    - `truediv`/`floordiv` is unambiguous, like Python `operator`s.
    """
    return {i:ti/a for i, ti in t.items()}

def tensitruediv(t:MutableMapping[tuple[int,...],Any], a:Any) -> MutableMapping[tuple[int,...],Any]:
    """True divide.
    
    $$
        t /= a
    $$
    """
    for i in t:
        t[i] /= a
    return t

def tensfloordiv(t:Mapping[tuple[int,...],Any], a:Any) -> dict[tuple[int,...],Any]:
    r"""Return the floor quotient.
    
    $$
        \left\lfloor\frac{t}{a}\right\rfloor
    $$
    """
    return {i:ti//a for i, ti in t.items()}

def tensifloordiv(t:MutableMapping[tuple[int,...],Any], a:Any) -> MutableMapping[tuple[int,...],Any]:
    """Floor divide.
    
    $$
        t //= a
    $$
    """
    for i in t:
        t[i] //= a
    return t

def tensmod(t:Mapping[tuple[int,...],Any], a:Any) -> dict[tuple[int,...],Any]:
    r"""Return the remainder.
    
    $$
        t \bmod a
    $$
    """
    return {i:ti%a for i, ti in t.items()}

def tensimod(t:MutableMapping[tuple[int,...],Any], a:Any) -> MutableMapping[tuple[int,...],Any]:
    r"""Mod.
    
    $$
        t \%= a
    $$
    """
    for i in t:
        t[i] %= a
    return t

def tensdivmod(t:Mapping[tuple[int,...],Any], a:Any) -> tuple[dict[tuple[int,...],Any], dict[tuple[int,...],Any]]:
    r"""Return the floor quotient and remainder.
    
    $$
        \left\lfloor\frac{t}{a}\right\rfloor, \ \left(t \bmod a\right)
    $$
    """
    q:dict[tuple[int,...],Any] = {}
    r:dict[tuple[int,...],Any] = {}
    for i, ti in t.items():
        q[i], r[i] = divmod(ti, a)
    return q, r
