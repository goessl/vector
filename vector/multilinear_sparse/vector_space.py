__all__ = ('tenspos', 'tensneg', 'tensadd', 'tensaddc', 'tenssub', 'tenssubc',
           'tensmul', 'tenstruediv', 'tensfloordiv', 'tensmod', 'tensdivmod')



def tenspos(t):
    """Return the tensor with the unary positive operator applied.
    
    $$
        +t
    $$
    """
    return {i:+ti for i, ti in t.items()}

def tensneg(t):
    """Return the tensor with the unary negative operator applied.
    
    $$
        -t
    $$
    """
    return {i:-ti for i, ti in t.items()}

def tensadd(*ts):
    r"""Return the sum of tensors.
    
    $$
        t_0 + t_1 + \cdots
    $$
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
    """Return `t` with `c` added to the `i`-th coefficient.
    
    $$
        t+ce_i
    $$
    
    More efficient than `tensadd(t, tensbasis(i, c))`.
    """
    r = dict(t)
    if i in r:
        r[i] += c
    else:
        r[i] = +c
    return r

def tenssub(s, t):
    """Return the difference of two tensors.
    
    $$
        s - t
    $$
    """
    r = dict(s)
    for i, ti in t.items():
        if i in r:
            r[i] -= ti
        else:
            r[i] = -ti
    return r

def tenssubc(t, c, i=()):
    """Return `t` with `c` subtracted from the `i`-th coefficient.
    
    $$
        t-ce_i
    $$
    
    More efficient than `tenssub(t, tensbasis(i, c))`.
    """
    r = dict(t)
    if i in r:
        r[i] -= c
    else:
        r[i] = -c
    return r

def tensmul(a, t):
    """Return the product of a scalar and a tensor.
    
    $$
        at
    $$
    """
    return {i:a*ti for i, ti in t.items()}

def tenstruediv(t, a):
    r"""Return the true division of a tensor and a scalar.
    
    $$
        \frac{t}{a}
    $$
    """
    return {i:ti/a for i, ti in t.items()}

def tensfloordiv(t, a):
    r"""Return the floor division of a tensor and a scalar.
    
    $$
        \left(\left\lfloor\frac{t_i}{a}\right\rfloor\right)_i
    $$
    """
    return {i:ti//a for i, ti in t.items()}

def tensmod(t, a):
    r"""Return the elementwise mod of a tensor and a scalar.
    
    $$
        \left(t_i \mod a\right)_i
    $$
    """
    return {i:ti%a for i, ti in t.items()}

def tensdivmod(t, a):
    r"""Return the elementwise divmod of a tensor and a scalar.
    
    $$
        \left(\left\lfloor\frac{t_i}{a}\right\rfloor\right)_i, \ \left(t_i \mod a\right)_i
    $$
    """
    q, r = {}, {}
    for i, ti in t.items():
        q[i], r[i] = divmod(ti, a)
    return q, r
