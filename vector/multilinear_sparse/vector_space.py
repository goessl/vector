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



def tenspos(t):
    """Return the identity.
    
    $$
        +t
    $$
    """
    return {i:+ti for i, ti in t.items()}

def tensipos(t):
    """Apply unary plus.
    
    $$
        t = +t
    $$
    """
    for i, ti in t.items():
        t[i] = +ti
    return t

def tensneg(t):
    """Return the negation.
    
    $$
        -t
    $$
    """
    return {i:-ti for i, ti in t.items()}

def tensineg(t):
    """Negate.
    
    $$
        t = -t
    $$
    """
    for i, ti in t.items():
        t[i] = -ti
    return t

def tensadd(*ts):
    r"""Return the sum.
    
    $$
        t_0 + t_1 + \cdots
    $$
    
    See also
    --------
    - for sum on a single coefficient: [`tensaddc`][vector.multilinear_sparse.vector_space.tensaddc]
    """
    r = dict(ts[0]) if ts else {}
    for t in ts[1:]:
        for i, ti in t.items():
            if i in r:
                r[i] += ti
            else:
                r[i] = +ti
    return r

def tensiadd(s, *ts):
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

def tensaddc(t, c, i=()):
    """Return the sum with a basis tensor.
    
    $$
        t + ce_i
    $$
    
    See also
    --------
    - for sum on more coefficients: [`tensadd`][vector.multilinear_sparse.vector_space.tensadd]
    """
    r = dict(t)
    if i in r:
        r[i] += c
    else:
        r[i] = +c
    return r

def tensiaddc(t, c, i=()):
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

def tenssub(s, t):
    """Return the difference.
    
    $$
        s - t
    $$
    
    See also
    --------
    - for difference on a single coefficient: [`tenssubc`][vector.multilinear_sparse.vector_space.tenssubc]
    """
    r = dict(s)
    for i, ti in t.items():
        if i in r:
            r[i] -= ti
        else:
            r[i] = -ti
    return r

def tensisub(s, t):
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

def tenssubc(t, c, i=()):
    """Return the difference with a basis tensor.
    
    $$
        t - ce_i
    $$
    
    See also
    --------
    - for difference on more coefficients: [`tenssub`][vector.multilinear_sparse.vector_space.tenssub]
    """
    r = dict(t)
    if i in r:
        r[i] -= c
    else:
        r[i] = -c
    return r

def tensisubc(t, c, i=()):
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

def tensmul(t, a):
    """Return the product.
    
    $$
        ta
    $$
    """
    return {i:ti*a for i, ti in t.items()}

def tensrmul(a, t):
    """Return the product.
    
    $$
        at
    $$
    """
    return {i:a*ti for i, ti in t.items()}

def tensimul(t, a):
    r"""Multiply.
    
    $$
        t \cdot= a
    $$
    """
    for i in t:
        t[i] *= a
    return t

def tenstruediv(t, a):
    r"""Return the true quotient.
    
    $$
        \frac{t}{a}
    $$
    """
    return {i:ti/a for i, ti in t.items()}

def tensitruediv(t, a):
    """True divide.
    
    $$
        t /= a
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
    for i in t:
        t[i] /= a
    return t

def tensfloordiv(t, a):
    r"""Return the floor quotient.
    
    $$
        \left\lfloor\frac{t}{a}\right\rfloor
    $$
    """
    return {i:ti//a for i, ti in t.items()}

def tensifloordiv(t, a):
    """Floor divide.
    
    $$
        t //= a
    $$
    """
    for i in t:
        t[i] //= a
    return t

def tensmod(t, a):
    r"""Return the remainder.
    
    $$
        t \bmod a
    $$
    """
    return {i:ti%a for i, ti in t.items()}

def tensimod(t, a):
    r"""Mod.
    
    $$
        t \%= a
    $$
    """
    for i in t:
        t[i] %= a
    return t

def tensdivmod(t, a):
    r"""Return the floor quotient and remainder.
    
    $$
        \left\lfloor\frac{t}{a}\right\rfloor, \ \left(t \bmod a\right)
    $$
    """
    q, r = {}, {}
    for i, ti in t.items():
        q[i], r[i] = divmod(ti, a)
    return q, r
