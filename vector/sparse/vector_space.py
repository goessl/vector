__all__ = ('vecspos', 'vecsneg', 'vecsadd', 'vecsaddc', 'vecssub', 'vecssubc',
           'vecsmul', 'vecsrmul', 'vecstruediv', 'vecsfloordiv', 'vecsmod', 'vecsdivmod')



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

def vecsadd(*vs):
    r"""Return the sum.
    
    $$
        \vec{v}_0+\vec{v}_1+\cdots
    $$
    
    Complexity
    ----------
    For two vectors with $n$ & $m$ elements there will be
    
    - $\min\{n, m\}$ scalar additions (`add`).
    
    See also
    --------
    - for sum on a single coefficient: [`vecsaddc`][vector.sparse.vector_space.vecsaddc]
    """
    r = dict(vs[0]) if vs else {}
    for v in vs[1:]:
        for i, vi in v.items():
            if i in r:
                r[i] += vi
            else:
                r[i] = vi
    return r

def vecsaddc(v, c, i=0):
    r"""Return the sum with a basis vector.
    
    $$
        \vec{v}+c\vec{e}_i
    $$
    
    Complexity
    ----------
    There will be
    
    - one scalar addition (`add`) if $i\in\vec{v}$ or
    - one unary plus operations (`pos`) otherwise.
    
    See also
    --------
    - for sum on more coefficients: [`vecsadd`][vector.sparse.vector_space.vecsadd]
    """
    r = dict(v)
    if i in r:
        r[i] += c
    else:
        r[i] = +c
    return r

def vecssub(v, w):
    r"""Return the difference.
    
    $$
        \vec{v}-\vec{w}
    $$
    
    Complexity
    ----------
    For two vectors with $n$ & $m$ elements there will be
    
    - $\min\{n, m\}$ scalar subtractions (`sub`) &
    - $\begin{cases}m-n&m\ge n\\0&m\le n\end{cases}$ negations (`neg`).
    
    See also
    --------
    - for difference on a single coefficient: [`vecssubc`][vector.sparse.vector_space.vecssubc]
    """
    r = dict(v)
    for i, wi in w.items():
        if i in r:
            r[i] -= wi
        else:
            r[i] = -wi
    return r

def vecssubc(v, c, i=0):
    r"""Return the difference with a basis vector.
    
    $$
        \vec{v}-c\vec{e}_i
    $$
    
    Complexity
    ----------
    There will be
    
    - one scalar subtraction (`sub`) if $i\in\vec{v}$ or
    - one scalar negation (`neg`) otherwise.
    
    See also
    --------
    - for difference on more coefficients: [`vecssub`][vector.sparse.vector_space.vecssub]
    """
    r = dict(v)
    if i in r:
        r[i] -= c
    else:
        r[i] = -c
    return r

def vecsmul(v, a):
    r"""Return the product.
    
    $$
        \vec{v}a
    $$
    
    Complexity
    ----------
    For a vector with $n$ elements there will be
    
    - $n$ scalar multiplications (`rmul`).
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
    
    - `div` would be more appropriate for an absolute clean mathematical
    implementation, that doesn't care about the language used. But the package
    might be used for pure integers/integer arithmetic, so both, `truediv`
    and `floordiv` operations have to be provided, and none should be
    privileged over the other by getting the universal `div` name.
    - `truediv`/`floordiv` is unambiguous, like Python `operator`s.
    """
    return {i:vi/a for i, vi in v.items()}

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
