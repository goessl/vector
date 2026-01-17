__all__ = ('tenspos', 'tensneg', 'tensadd', 'tensaddc', 'tenssub', 'tenssubc',
           'tensmul', 'tensrmul', 'tenstruediv', 'tensfloordiv', 'tensmod', 'tensdivmod')



def tenspos(t):
    """Return the identity.
    
    $$
        +t
    $$
    """
    return {i:+ti for i, ti in t.items()}

def tensneg(t):
    """Return the negation.
    
    $$
        -t
    $$
    """
    return {i:-ti for i, ti in t.items()}

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
                r[i] = ti
    return r

def tensaddc(t, c, i=()):
    """Return the sum with a basis tensor
    
    $$
        t+ce_i
    $$
    
    More efficient than `tensadd(t, tensbasis(i, c))`.
    
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

def tenssubc(t, c, i=()):
    """Return the difference with a basis tensor.
    
    $$
        t-ce_i
    $$
    
    More efficient than `tenssub(t, tensbasis(i, c))`.
    
    See also
    --------
    - for difference on more coefficients: [`tensub`][vector.multilinear_sparse.vector_space.tenssub]
    """
    r = dict(t)
    if i in r:
        r[i] -= c
    else:
        r[i] = -c
    return r

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

def tenstruediv(t, a):
    r"""Return the true quotient.
    
    $$
        \frac{t}{a}
    $$
    """
    return {i:ti/a for i, ti in t.items()}

def tensfloordiv(t, a):
    r"""Return the floor quotient.
    
    $$
        \left\lfloor\frac{t}{a}\right\rfloor
    $$
    """
    return {i:ti//a for i, ti in t.items()}

def tensmod(t, a):
    r"""Return the remainder.
    
    $$
        t \bmod a
    $$
    """
    return {i:ti%a for i, ti in t.items()}

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
