__all__ = ('vecsmap', 'vecspos', 'vecsneg', 'vecsadd', 'vecsaddc', 'vecssub', 'vecssubc',
           'vecsmul', 'vecstruediv', 'vecsfloordiv', 'vecsmod', 'vecsdivmod')


def vecsmap(f, v):
    r"""Return the vector with the function `f` applied elementwise.
    
    $$
        \left(f(v_i)\right_i
    $$
    
    Complexity
    ----------
    For a vector of $n$ elements there will be
    
    - $n$ scalar calls to `f`.
    """
    return {i:f(vi) for i, vi in v.items()}

def vecspos(v):
    r"""Return the vector with the unary positive operator applied.
    
    $$
        +\vec{v} \qquad \mathbb{K}^n\to\mathbb{K}^n
    $$
    
    Complexity
    ----------
    For a vector with $n$ elements there will be
    
    - $n$ scalar unary plus operations (`pos`).
    """
    return {i:+vi for i, vi in v.items()}

def vecsneg(v):
    r"""Return the vector with the unary negative operator applied.
    
    $$
        -\vec{v} \qquad \mathbb{K}^n\to\mathbb{K}^n
    $$
    
    Complexity
    ----------
    For a vector with $n$ elements there will be
    
    - $n$ scalar negations (`neg`).
    """
    return {i:-vi for i, vi in v.items()}

def vecsadd(*vs):
    r"""Return the sum of vectors.
    
    $$
        \vec{v}_0+\vec{v}_1+\cdots \qquad \mathbb{K}^{n_0}\times\mathbb{K}^{n_1}\times\cdots\to\mathbb{K}^{\max_i n_i}
    $$
    
    Complexity
    ----------
    For two vectors with $n$ & $m$ elements there will be
    
    - $\min\{n, m\}$ scalar additions (`add`).
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
    r"""Return `v` with `c` added to the `i`-th coefficient.
    
    $$
        \vec{v}+c\vec{e}_i \qquad \mathbb{K}^n\to\mathbb{K}^{\max\{n, i\}}
    $$
    
    Complexity
    ----------
    There will be
    
    - one scalar addition (`add`) if $i\in\vec{v}$ or
    - one unary plus operations (`pos`) otherwise.
    """
    r = dict(v)
    if i in r:
        r[i] += c
    else:
        r[i] = +c
    return r

def vecssub(v, w):
    r"""Return the difference of two vectors.
    
    $$
        \vec{v}-\vec{w} \qquad \mathbb{K}^m\times\mathbb{K}^n\to\mathbb{K}^{\max\{m, n\}}
    $$
    
    Complexity
    ----------
    For two vectors with $n$ & $m$ elements there will be
    
    - $\min\{n, m\}$ scalar subtractions (`sub`) &
    - $\begin{cases}m-n&m\ge n\\0&m\le n\end{cases}$ negations (`neg`).
    """
    r = dict(v)
    for i, wi in w.items():
        if i in r:
            r[i] -= wi
        else:
            r[i] = -wi
    return r

def vecssubc(v, c, i=0):
    r"""Return `v` with `c` added to the `i`-th coefficient.
    
    $$
        \vec{v}-c\vec{e}_i \qquad \mathbb{K}^n\to\mathbb{K}^{\max\{n, i\}}
    $$
    
    Complexity
    ----------
    There will be
    
    - one scalar subtraction (`sub`) if $i\in\vec{v}$ or
    - one scalar negation (`neg`) otherwise.
    """
    r = dict(v)
    if i in r:
        r[i] -= c
    else:
        r[i] = -c
    return r

def vecsmul(a, v):
    r"""Return the product of a scalar and a vector.
    
    $$
        a\vec{v} \qquad \mathbb{K}\times\mathbb{K}^n\to\mathbb{K}^n
    $$
    
    Complexity
    ----------
    For a vector with $n$ elements there will be
    
    - $n$ scalar multiplications (`rmul`).
    """
    return {i:a*vi for i, vi in v.items()}

def vecstruediv(v, a):
    r"""Return the true division of a vector and a scalar.
    
    $$
        \frac{\vec{v}}{a} \qquad \mathbb{K}^n\times\mathbb{K}\to\mathbb{K}^n
    $$
    
    Complexity
    ----------
    For a vector with $n$ elements there will be
    
    - $n$ scalar true divisions (`truediv`).
    """
    return {i:vi/a for i, vi in v.items()}

def vecsfloordiv(v, a):
    r"""Return the floor division of a vector and a scalar.
    
    $$
        \left(\left\lfloor\frac{v_i}{a}\right\rfloor\right)_i \qquad \mathbb{K}^n\times\mathbb{K}\to\mathbb{K}^n
    $$
    
    Complexity
    ----------
    For a vector with $n$ elements there will be
    
    - $n$ scalar floor divisions (`floordiv`).
    """
    return {i:vi//a for i, vi in v.items()}

def vecsmod(v, a):
    r"""Return the elementwise mod of a vector and a scalar.
    
    $$
        \left(v_i \mod a\right)_i \qquad \mathbb{K}^n\times\mathbb{K}\to\mathbb{K}^n
    $$
    
    Complexity
    ----------
    For a vector with $n$ elements there will be
    
    - $n$ scalar modulos (`mod`).
    """
    return {i:vi%a for i, vi in v.items()}

def vecsdivmod(v, a):
    r"""Return the elementwise divmod of a vector and a scalar.
    
    $$
        \left(\left\lfloor\frac{v_i}{a}\right\rfloor\right)_i, \ \left(v_i \mod a\right)_i \qquad \mathbb{K}^n\times\mathbb{K}\to\mathbb{K}^n\times\mathbb{K}^n
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
