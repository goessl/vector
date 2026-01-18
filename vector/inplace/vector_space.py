__all__ = ('vecipos', 'vecineg', 'veciadd', 'veciaddc', 'vecisub', 'vecisubc',
           'vecimul', 'vecitruediv', 'vecifloordiv', 'vecimod')



def vecipos(v):
    r"""Apply unary positive.
    
    $$
        \vec{v} = +\vec{v} \qquad \mathbb{K}^n\to\mathbb{K}^n
    $$
    """
    for i, vi in enumerate(v):
        v[i] = +vi
    return v

def vecineg(v):
    r"""Negate.
    
    $$
        \vec{v} = -\vec{v} \qquad \mathbb{K}^n\to\mathbb{K}^n
    $$
    """
    for i, vi in enumerate(v):
        v[i] = -vi
    return v

def veciadd(v, *ws):
    r"""Add.
    
    $$
        \vec{v} += \vec{w}_0+\vec{w}_1+\cdots \qquad \mathbb{K}^{n_0}\times\mathbb{K}^{n_1}\times\cdots\to\mathbb{K}^{\max_i n_i}
    $$
    
    See also
    --------
    - for sum on a single coefficient: [`veciaddc`][vector.inplace.vector_space.veciaddc]
    """
    if ws:
        #extract longest vector so we only have to extend once
        longest = max(ws, key=len)
        ws = tuple(w for w in ws if w is not longest)
        #add longest vector
        for i, wi in enumerate(longest[:len(v)]):
            v[i] += wi
        v.extend(longest[len(v):])
        #add others
        for w in ws:
            for i, wi in enumerate(w):
                v[i] += wi
    return v

def veciaddc(v, c, i=0, zero=0):
    r"""Add a basis vector.
    
    $$
        \vec{v} += c\vec{e}_i \qquad \mathbb{K}^n\to\mathbb{K}^{\max\{n, i+1\}}
    $$
    
    More efficient than `veciadd(v, vecibasis(i, c))`.
    
    See also
    --------
    - for sum on more coefficients: [`veciadd`][vector.inplace.vector_space.veciadd]
    """
    if i >= len(v):
        v.extend([zero] * (i-len(v)))
        v.append(c)
    else:
        v[i] += c
    return v

def vecisub(v, w):
    r"""Subtract.
    
    $$
        \vec{v} -= \vec{w} \qquad \mathbb{K}^m\times\mathbb{K}^n\to\mathbb{K}^{\max\{m, n\}}
    $$
    
    See also
    --------
    - for difference on a single coefficient: [`vecisubc`][vector.inplace.vector_space.vecisubc]
    """
    for i, wi in enumerate(w[:len(v)]):
        v[i] -= wi
    if len(w) > len(v):
        v.extend([-wi for wi in w[len(v):]])
    return v

def vecisubc(v, c, i=0, zero=0):
    r"""Subtract a basis vector.
    
    $$
        \vec{v} -= c\vec{e}_i \qquad \mathbb{K}^n\to\mathbb{K}^{\max\{n, i+1\}}
    $$
    
    More efficient than `vecisub(v, vecibasis(i, c))`.
    
    See also
    --------
    - for difference on more coefficients: [`vecisub`][vector.inplace.vector_space.vecisub]
    """
    if i >= len(v):
        v.extend([zero] * (i-len(v)))
        v.append(-c)
    else:
        v[i] -= c
    return v

def vecimul(v, a):
    r"""Multiply.
    
    $$
        \vec{v} \cdot= a \qquad \mathbb{K}\times\mathbb{K}^n\to\mathbb{K}^n
    $$
    """
    for i in range(len(v)):
        v[i] *= a
    return v

def vecitruediv(v, a):
    r"""True divide.
    
    $$
        \vec{v} /= a \qquad \mathbb{K}^n\times\mathbb{K}\to\mathbb{K}^n
    $$
    
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
    for i in range(len(v)):
        v[i] /= a
    return v

def vecifloordiv(v, a):
    r"""Floor divide.
    
    $$
        \vec{v} //= a \qquad \mathbb{K}^n\times\mathbb{K}\to\mathbb{K}^n
    $$
    """
    for i in range(len(v)):
        v[i] //= a
    return v

def vecimod(v, a):
    r"""Mod.
    
    $$
        \vec{v} \%= a \qquad \mathbb{K}^n\times\mathbb{K}\to\mathbb{K}^n
    $$
    """
    for i in range(len(v)):
        v[i] %= a
    return v
