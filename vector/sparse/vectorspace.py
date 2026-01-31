__all__ = ('vecspos',             'vecsipos',
           'vecsneg',             'vecsineg',
           'vecsadd',             'vecsiadd',
           'vecsaddc',            'vecsiaddc',
           'vecssub',             'vecsisub',
           'vecssubc',            'vecsisubc',
           'vecsmul', 'vecsrmul', 'vecsimul',
           'vecstruediv',         'vecsitruediv',
           'vecsfloordiv',        'vecsifloordiv',
           'vecsmod',             'vecsimod',
           'vecsdivmod')



def vecspos(v):
    r"""Return the identity.
    
    $$
        +\vec{v}
    $$
    
    Complexity
    ----------
    For a vector with $n$ elements there will be
    
    - $n$ scalar unary plus operations (`pos`).
    """
    return {i:+vi for i, vi in v.items()}

def vecsipos(v):
    r"""Apply unary plus.
    
    $$
        \vec{v} = +\vec{v}
    $$
    
    Complexity
    ----------
    For a vector with $n$ elements there will be
    
    - $n$ scalar unary plus operations (`pos`).
    """
    for i, vi in v.items():
        v[i] = +vi
    return v

def vecsneg(v):
    r"""Return the negation.
    
    $$
        -\vec{v}
    $$
    
    Complexity
    ----------
    For a vector with $n$ elements there will be
    
    - $n$ scalar negations (`neg`).
    """
    return {i:-vi for i, vi in v.items()}

def vecsineg(v):
    r"""Negate.
    
    $$
        \vec{v} = -\vec{v}
    $$
    
    Complexity
    ----------
    For a vector with $n$ elements there will be
    
    - $n$ scalar negations (`neg`).
    """
    for i, vi in v.items():
        v[i] = -vi
    return v

def vecsadd(*vs):
    r"""Return the sum.
    
    $$
        \vec{v}_0+\vec{v}_1+\cdots
    $$
    
    Complexity
    ----------
    For two vectors with $n$ & $m$ elements there will be
    
    - $\min\{n, m\}$ scalar additions (`iadd`) &
    - $\begin{cases}m-n&m\ge n\\0&m\le n\end{cases}$ unary plus operations (`pos`).
    
    See also
    --------
    - for sum on a single coefficient: [`vecsaddc`][vector.sparse.vectorspace.vecsaddc]
    """
    r = dict(vs[0]) if vs else {}
    for v in vs[1:]:
        for i, vi in v.items():
            if i in r:
                r[i] += vi
            else:
                r[i] = +vi
    return r

def vecsiadd(v, *ws):
    r"""Add.
    
    $$
        \vec{v} += \vec{w}_0+\vec{w}_1+\cdots
    $$
    
    Complexity
    ----------
    For two vectors with $n$ & $m$ elements there will be
    
    - $\min\{n, m\}$ scalar additions (`iadd`) &
    - $\begin{cases}m-n&m\ge n\\0&m\le n\end{cases}$ unary plus operations (`pos`).
    
    See also
    --------
    - for sum on a single coefficient: [`vecsiaddc`][vector.sparse.vectorspace.vecsiaddc]
    """
    for w in ws:
        for i, wi in w.items():
            if i in v:
                v[i] += wi
            else:
                v[i] = +wi
    return v

def vecsaddc(v, c, i=0):
    r"""Return the sum with a basis vector.
    
    $$
        \vec{v}+c\vec{e}_i
    $$
    
    Complexity
    ----------
    There will be
    
    - one scalar addition (`iadd`) if $i\in\vec{v}$ or
    - one unary plus operation (`pos`) otherwise.
    
    See also
    --------
    - for sum on more coefficients: [`vecsadd`][vector.sparse.vectorspace.vecsadd]
    """
    r = dict(v)
    if i in r:
        r[i] += c
    else:
        r[i] = +c
    return r

def vecsiaddc(v, c, i=0):
    r"""Add a basis vector.
    
    $$
        \vec{v} += c\vec{e}_i
    $$
    
    Complexity
    ----------
    There will be
    
    - one scalar addition (`iadd`) if $i\in\vec{v}$ or
    - one unary plus operation (`pos`) otherwise.
    
    See also
    --------
    - for sum on more coefficients: [`vecsiadd`][vector.sparse.vectorspace.vecsiadd]
    """
    if i in v:
        v[i] += c
    else:
        v[i] = +c
    return v

def vecssub(v, w):
    r"""Return the difference.
    
    $$
        \vec{v}-\vec{w}
    $$
    
    Complexity
    ----------
    For two vectors with $n$ & $m$ elements there will be
    
    - $\min\{n, m\}$ scalar subtractions (`isub`) &
    - $\begin{cases}m-n&m\ge n\\0&m\le n\end{cases}$ negations (`neg`).
    
    See also
    --------
    - for difference on a single coefficient: [`vecssubc`][vector.sparse.vectorspace.vecssubc]
    """
    r = dict(v)
    for i, wi in w.items():
        if i in r:
            r[i] -= wi
        else:
            r[i] = -wi
    return r

def vecsisub(v, w):
    r"""Subtract.
    
    $$
        \vec{v} -= \vec{w}
    $$
    
    Complexity
    ----------
    For two vectors with $n$ & $m$ elements there will be
    
    - $\min\{n, m\}$ scalar subtractions (`isub`) &
    - $\begin{cases}m-n&m\ge n\\0&m\le n\end{cases}$ negations (`neg`).
    
    See also
    --------
    - for difference on a single coefficient: [`vecsisubc`][vector.sparse.vectorspace.vecsisubc]
    """
    for i, wi in w.items():
        if i in v:
            v[i] -= wi
        else:
            v[i] = -wi
    return v

def vecssubc(v, c, i=0):
    r"""Return the difference with a basis vector.
    
    $$
        \vec{v}-c\vec{e}_i
    $$
    
    Complexity
    ----------
    There will be
    
    - one scalar subtraction (`isub`) if $i\in\vec{v}$ or
    - one scalar negation (`neg`) otherwise.
    
    See also
    --------
    - for difference on more coefficients: [`vecssub`][vector.sparse.vectorspace.vecssub]
    """
    r = dict(v)
    if i in r:
        r[i] -= c
    else:
        r[i] = -c
    return r

def vecsisubc(v, c, i=0):
    r"""Subtract a basis vector.
    
    $$
        \vec{v} -= c\vec{e}_i
    $$
    
    Complexity
    ----------
    There will be
    
    - one scalar subtraction (`isub`) if $i\in\vec{v}$ or
    - one scalar negation (`neg`) otherwise.
    
    See also
    --------
    - for difference on more coefficients: [`vecsisub`][vector.sparse.vectorspace.vecsisub]
    """
    if i in v:
        v[i] -= c
    else:
        v[i] = -c
    return v

def vecsmul(v, a):
    r"""Return the product.
    
    $$
        \vec{v}a
    $$
    
    Complexity
    ----------
    For a vector with $n$ elements there will be
    
    - $n$ scalar multiplications (`mul`).
    """
    return {i:vi*a for i, vi in v.items()}

def vecsrmul(a, v):
    r"""Return the product.
    
    $$
        a\vec{v}
    $$
    
    Complexity
    ----------
    For a vector with $n$ elements there will be
    
    - $n$ scalar multiplications (`rmul`).
    """
    return {i:a*vi for i, vi in v.items()}

def vecsimul(v, a):
    r"""Multiply.
    
    $$
        \vec{v} \cdot= a
    $$
    
    Complexity
    ----------
    For a vector with $n$ elements there will be
    
    - $n$ scalar multiplications (`imul`).
    """
    for i in v:
        v[i] *= a
    return v

def vecstruediv(v, a):
    r"""Return the true quotient.
    
    $$
        \frac{\vec{v}}{a}
    $$
    
    Complexity
    ----------
    For a vector with $n$ elements there will be
    
    - $n$ scalar true divisions (`truediv`).
    
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
    return {i:vi/a for i, vi in v.items()}

def vecsitruediv(v, a):
    r"""True divide.
    
    $$
        \vec{v} /= a
    $$
    
    Complexity
    ----------
    For a vector with $n$ elements there will be
    
    - $n$ scalar true divisions (`itruediv`).
    
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
    for i in v:
        v[i] /= a
    return v

def vecsfloordiv(v, a):
    r"""Return the floor quotient.
    
    $$
        \left\lfloor\frac{\vec{v}}{a}\right\rfloor
    $$
    
    Complexity
    ----------
    For a vector with $n$ elements there will be
    
    - $n$ scalar floor divisions (`floordiv`).
    """
    return {i:vi//a for i, vi in v.items()}

def vecsifloordiv(v, a):
    r"""Floor divide.
    
    $$
        \vec{v} //= a
    $$
    
    Complexity
    ----------
    For a vector with $n$ elements there will be
    
    - $n$ scalar floor divisions (`ifloordiv`).
    """
    for i in v:
        v[i] //= a
    return v

def vecsmod(v, a):
    r"""Return the remainder.
    
    $$
        \vec{v} \bmod a
    $$
    
    Complexity
    ----------
    For a vector with $n$ elements there will be
    
    - $n$ scalar modulos (`mod`).
    """
    return {i:vi%a for i, vi in v.items()}

def vecsimod(v, a):
    r"""Mod.
    
    $$
        \vec{v} \%= a
    $$
    
    Complexity
    ----------
    For a vector with $n$ elements there will be
    
    - $n$ scalar modulos (`imod`).
    """
    for i in v:
        v[i] %= a
    return v

def vecsdivmod(v, a):
    r"""Return the floor quotient and remainder.
    
    $$
        \left\lfloor\frac{\vec{v}}{a}\right\rfloor, \ \left(\vec{v} \bmod a\right)
    $$
    
    Complexity
    ----------
    For a vector with $n$ elements there will be
    
    - $n$ scalar divmods (`divmod`).
    """
    q, r = {}, {}
    for i, vi in v.items():
        q[i], r[i] = divmod(vi, a)
    return q, r
