from ..lazy import veclpos, veclneg, vecladd, vecladdc, veclsub, veclsubc, veclmul, vecltruediv, veclfloordiv, veclmod



__all__ = ('vecpos', 'vecneg', 'vecadd', 'vecaddc', 'vecsub', 'vecsubc',
           'vecmul', 'vectruediv', 'vecfloordiv', 'vecmod', 'vecdivmod')



def vecpos(v):
    r"""Return the vector with the unary positive operator applied.
    
    $$
        +\vec{v} \qquad \mathbb{K}^n\to\mathbb{K}^n
    $$
    
    Complexity
    ----------
    For a vector of length $n$ there will be
    
    - $n$ scalar unary plus operations (`pos`).
    """
    return tuple(veclpos(v))

def vecneg(v):
    r"""Return the vector with the unary negative operator applied.
    
    $$
        -\vec{v} \qquad \mathbb{K}^n\to\mathbb{K}^n
    $$
    
    Complexity
    ----------
    For a vector of length $n$ there will be
    
    - $n$ scalar negations (`neg`).
    """
    return tuple(veclneg(v))

def vecadd(*vs):
    r"""Return the sum of vectors.
    
    $$
        \vec{v}_0+\vec{v}_1+\cdots \qquad \mathbb{K}^{n_0}\times\mathbb{K}^{n_1}\times\cdots\to\mathbb{K}^{\max_i n_i}
    $$
    
    Complexity
    ----------
    For two vectors of lengths $n$ & $m$ there will be
    
    - $\min\{n, m\}$ scalar additions (`add`).
    """
    return tuple(vecladd(*vs))

def vecaddc(v, c, i=0, zero=0):
    r"""Return `v` with `c` added to the `i`-th coefficient.
    
    $$
        \vec{v}+c\vec{e}_i \qquad \mathbb{K}^n\to\mathbb{K}^{\max\{n, i\}}
    $$
    
    More efficient than `vecadd(v, vecbasis(i, c))`.
    
    Complexity
    ----------
    There will be
    
    - one scalar addition (`add`) if $i\le n$ or
    - one unary plus operations (`pos`) otherwise.
    """
    return tuple(vecladdc(v, c, i=i, zero=zero))

def vecsub(v, w):
    r"""Return the difference of two vectors.
    
    $$
        \vec{v}-\vec{w} \qquad \mathbb{K}^m\times\mathbb{K}^n\to\mathbb{K}^{\max\{m, n\}}
    $$
    
    Complexity
    ----------
    For two vectors of lengths $n$ & $m$ there will be
    
    - $\min\{n, m\}$ scalar subtractions (`sub`) &
    - $\begin{cases}m-n&m\ge n\\0&m\le n\end{cases}$ negations (`neg`).
    """
    return tuple(veclsub(v, w))

def vecsubc(v, c, i=0, zero=0):
    r"""Return `v` with `c` added to the `i`-th coefficient.
    
    $$
        \vec{v}-c\vec{e}_i \qquad \mathbb{K}^n\to\mathbb{K}^{\max\{n, i\}}
    $$
    
    More efficient than `vecsub(v, vecbasis(i, c))`.
    
    Complexity
    ----------
    There will be
    
    - one scalar subtraction (`sub`) if $i\le n$ or
    - one scalar negation (`neg`) otherwise.
    """
    return tuple(veclsubc(v, c, i=i, zero=zero))

def vecmul(a, v):
    r"""Return the product of a scalar and a vector.
    
    $$
        a\vec{v} \qquad \mathbb{K}\times\mathbb{K}^n\to\mathbb{K}^n
    $$
    
    Complexity
    ----------
    For a vector of length $n$ there will be
    
    - $n$ scalar multiplications (`rmul`).
    """
    return tuple(veclmul(a, v))

def vectruediv(v, a):
    r"""Return the true division of a vector and a scalar.
    
    $$
        \frac{\vec{v}}{a} \qquad \mathbb{K}^n\times\mathbb{K}\to\mathbb{K}^n
    $$
    
    Complexity
    ----------
    For a vector of length $n$ there will be
    
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
    return tuple(vecltruediv(v, a))

def vecfloordiv(v, a):
    r"""Return the floor division of a vector and a scalar.
    
    $$
        \left(\left\lfloor\frac{v_i}{a}\right\rfloor\right)_i \qquad \mathbb{K}^n\times\mathbb{K}\to\mathbb{K}^n
    $$
    
    Complexity
    ----------
    For a vector of length $n$ there will be
    
    - $n$ scalar floor divisions (`floordiv`).
    """
    return tuple(veclfloordiv(v, a))

def vecmod(v, a):
    r"""Return the elementwise mod of a vector and a scalar.
    
    $$
        \left(v_i \mod a\right)_i \qquad \mathbb{K}^n\times\mathbb{K}\to\mathbb{K}^n
    $$
    
    Complexity
    ----------
    For a vector of length $n$ there will be
    
    - $n$ scalar modulos (`mod`).
    """
    return tuple(veclmod(v, a))

def vecdivmod(v, a):
    r"""Return the elementwise divmod of a vector and a scalar.
    
    $$
        \left(\left\lfloor\frac{v_i}{a}\right\rfloor\right)_i, \ \left(v_i \mod a\right)_i \qquad \mathbb{K}^n\times\mathbb{K}\to\mathbb{K}^n\times\mathbb{K}^n
    $$
    
    Complexity
    ----------
    For a vector of length $n$ there will be
    
    - $n$ scalar divmods (`divmod`).
    """
    q, r = [], []
    for vi in v:
        qi, ri = divmod(vi, a)
        q.append(qi)
        r.append(ri)
    return tuple(q), tuple(r)
