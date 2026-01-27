from typing import TypeVar



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



T = TypeVar('T')
Index = tuple[int, ...]



def tenspos(t:dict[Index,T]) -> dict[Index,T]:
    """Return the identity.
    
    $$
        +t
    $$
    """
    return {i:+ti for i, ti in t.items()}

def tensipos(t:dict[Index,T]) -> dict[Index,T]:
    """Apply unary plus.
    
    $$
        t = +t
    $$
    """
    for i, ti in t.items():
        t[i] = +ti
    return t

def tensneg(t:dict[Index,T]) -> dict[Index,T]:
    """Return the negation.
    
    $$
        -t
    $$
    """
    return {i:-ti for i, ti in t.items()}

def tensineg(t:dict[Index,T]) -> dict[Index,T]:
    """Negate.
    
    $$
        t = -t
    $$
    """
    for i, ti in t.items():
        t[i] = -ti
    return t

def tensadd(*ts:dict[Index,T]) -> dict[Index,T]:
    r"""Return the sum.
    
    $$
        t_0 + t_1 + \cdots
    $$
    
    See also
    --------
    - for sum on a single coefficient: [`tensaddc`][vector.multilinear_sparse.vector_space.tensaddc]
    """
    r:dict[Index,T] = dict(ts[0]) if ts else {}
    for t in ts[1:]:
        for i, ti in t.items():
            if i in r:
                r[i] += ti
            else:
                r[i] = +ti
    return r

def tensiadd(s:dict[Index,T], *ts:dict[Index,T]) -> dict[Index,T]:
    r"""Add.
    
    $$
        s += t_0 + t_1 + \cdots
    $$
    
    See also
    --------
    - for sum on a single coefficient: [`tensiaddc`][vector.multilinear_sparse.vector_space.tensiaddc]
    """
    for t in ts:
        for i, ti in t.items():
            if i in s:
                s[i] += ti
            else:
                s[i] = +ti
    return s

def tensaddc(t:dict[Index,T], c:T, i:Index=()) -> dict[Index,T]:
    """Return the sum with a basis tensor.
    
    $$
        t + ce_i
    $$
    
    See also
    --------
    - for sum on more coefficients: [`tensadd`][vector.multilinear_sparse.vector_space.tensadd]
    """
    r:dict[Index,T] = dict(t)
    if i in r:
        r[i] += c
    else:
        r[i] = +c
    return r

def tensiaddc(t:dict[Index,T], c:T, i:Index=()) -> dict[Index,T]:
    """Add a basis tensor.
    
    $$
        t += ce_i
    $$
    
    See also
    --------
    - for sum on more coefficients: [`tensiadd`][vector.multilinear_sparse.vector_space.tensiadd]
    """
    if i in t:
        t[i] += c
    else:
        t[i] = +c
    return t

def tenssub(s:dict[Index,T], t:dict[Index,T]) -> dict[Index,T]:
    """Return the difference.
    
    $$
        s - t
    $$
    
    See also
    --------
    - for difference on a single coefficient: [`tenssubc`][vector.multilinear_sparse.vector_space.tenssubc]
    """
    r:dict[Index,T] = dict(s)
    for i, ti in t.items():
        if i in r:
            r[i] -= ti
        else:
            r[i] = -ti
    return r

def tensisub(s:dict[Index,T], t:dict[Index,T]) -> dict[Index,T]:
    """Subtract.
    
    $$
        s -= t
    $$
    
    See also
    --------
    - for difference on a single coefficient: [`tensisubc`][vector.multilinear_sparse.vector_space.tensisubc]
    """
    for i, ti in t.items():
        if i in s:
            s[i] -= ti
        else:
            s[i] = -ti
    return s

def tenssubc(t:dict[Index,T], c:T, i:Index=()) -> dict[Index,T]:
    """Return the difference with a basis tensor.
    
    $$
        t - ce_i
    $$
    
    See also
    --------
    - for difference on more coefficients: [`tenssub`][vector.multilinear_sparse.vector_space.tenssub]
    """
    r:dict[Index,T] = dict(t)
    if i in r:
        r[i] -= c
    else:
        r[i] = -c
    return r

def tensisubc(t:dict[Index,T], c:T, i:Index=()) -> dict[Index,T]:
    """Subtract a basis tensor.
    
    $$
        t -= ce_i
    $$
    
    See also
    --------
    - for difference on more coefficients: [`tensisub`][vector.multilinear_sparse.vector_space.tensisub]
    """
    if i in t:
        t[i] -= c
    else:
        t[i] = -c
    return t

def tensmul(t:dict[Index,T], a:T) -> dict[Index,T]:
    """Return the product.
    
    $$
        ta
    $$
    """
    return {i:ti*a for i, ti in t.items()}

def tensrmul(a:T, t:dict[Index,T]) -> dict[Index,T]:
    """Return the product.
    
    $$
        at
    $$
    """
    return {i:a*ti for i, ti in t.items()}

def tensimul(t:dict[Index,T], a:T) -> dict[Index,T]:
    r"""Multiply.
    
    $$
        t \cdot= a
    $$
    """
    for i in t:
        t[i] *= a
    return t

def tenstruediv(t:dict[Index,T], a:T) -> dict[Index,T]:
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

def tensitruediv(t:dict[Index,T], a:T) -> dict[Index,T]:
    """True divide.
    
    $$
        t /= a
    $$
    """
    for i in t:
        t[i] /= a
    return t

def tensfloordiv(t:dict[Index,T], a:T) -> dict[Index,T]:
    r"""Return the floor quotient.
    
    $$
        \left\lfloor\frac{t}{a}\right\rfloor
    $$
    """
    return {i:ti//a for i, ti in t.items()}

def tensifloordiv(t:dict[Index,T], a:T) -> dict[Index,T]:
    """Floor divide.
    
    $$
        t //= a
    $$
    """
    for i in t:
        t[i] //= a
    return t

def tensmod(t:dict[Index,T], a:T) -> dict[Index,T]:
    r"""Return the remainder.
    
    $$
        t \bmod a
    $$
    """
    return {i:ti%a for i, ti in t.items()}

def tensimod(t:dict[Index,T], a:T) -> dict[Index,T]:
    r"""Mod.
    
    $$
        t \%= a
    $$
    """
    for i in t:
        t[i] %= a
    return t

def tensdivmod(t:dict[Index,T], a:T) -> tuple[dict[Index,T], dict[Index,T]]:
    r"""Return the floor quotient and remainder.
    
    $$
        \left\lfloor\frac{t}{a}\right\rfloor, \ \left(t \bmod a\right)
    $$
    """
    q:dict[Index,T] = {}
    r:dict[Index,T] = {}
    for i, ti in t.items():
        q[i], r[i] = divmod(ti, a)
    return q, r
