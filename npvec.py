import numpy as np
from itertools import zip_longest
from vector import veceq



#creation stuff
def vecnpzero(d=None):
    """Return `d` zero vectors.
    
    The retured value is a `(d, 1)`-array of zeros if `d` is not `None`
    or `[0]` otherwise.
    """
    #same dtype as numpy.polynomial.polynomial.polyzero
    return np.zeros(1 if d is None else (d, 1), dtype=np.int64)

def vecnpbasis(i, c=1, d=None):
    """Return `d` many `i`-th basis vectors times `c`.
    
    The retured value is a `(d, i+1)`-array if `d` is not `None`
    or `(i+1,)` otherwise.
    """
    #choose dtype acc to c
    v = np.zeros(i+1 if d is None else (d, i+1), dtype=np.asarray(c).dtype)
    v[..., -1] = c #maybe scalar, maybe (d,)-array
    return v

def vecnprand(n, d=None):
    """Return `d` random vectors of `n` uniform coefficients in `[0, 1[`.
    
    The retured value is a `(d, n)`-array if `d` is not `None`
    or `(n,)` otherwise.
    """
    return np.random.rand(*((n,) if d is None else (d, n)))

def vecnprandn(n, normed=True, d=None):
    """Return `d` random vectors of `n` normal distributed coefficients.
    
    The retured value is a `(d, n)`-array if `d` is not `None`
    or `(n,)` otherwise.
    """
    v = np.random.randn(*((n,) if d is None else (d, n)))
    return v/np.linalg.norm(v, axis=-1, keepdims=True) if normed else v


#sequence stuff
def vecnpeq(v, w):
    """Return if two vectors are equal."""
    v, w = np.asarray(v), np.asarray(w)
    if v.ndim == w.ndim == 1:
        return veceq(v, w)
    return np.array([veceq(v[i] if v.ndim>1 else v, w[i] if w.ndim>1 else w) \
            for i in range(v.shape[0] if v.ndim>1 else w.shape[0])])

def vecnptrim(v, tol=1e-9):
    """Remove all trailing near zero (abs(v_i)<=tol) coefficients."""
    v = np.asarray(v)
    #use np.all, because v[...,-1] maybe a scalar (not iterable)
    while v.shape[-1]>1 and np.all(abs(v[...,-1])<=tol):
        v = v[...,:-1]
    if v.shape[-1]==1 and np.all(abs(v[...,0])<=tol): #leave 'leading' zero
        v[...,0] = 0
    return v

#vecnpround = np.round


#Hilbert space stuff
def vecnpabsq(v):
    """Return the sum of absolute squares of the coefficients."""
    return np.sum(np.abs(v)**2, axis=-1)

def vecnpabs(v):
    """Return the Euclidean/L2-norm."""
    return np.linalg.norm(v, axis=-1)

def vecnpdot(v, w):
    """Return the inner product of two vectors without conjugation."""
    v, w = np.asarray(v), np.asarray(w)
    shape = tuple(reversed(tuple(
            map(min, zip(reversed(v.shape), reversed(w.shape))))))
    return np.sum(v[...,*map(slice, shape)]*w[...,*map(slice, shape)], axis=-1)


#vector space stuff
def vecnpadd(*vs):
    """Return the sum of vectors."""
    if not vs: #empty sum
        return vecnpzero()
    vs = tuple(np.asarray(v) for v in vs)
    shape = tuple(reversed(tuple(map(max, zip_longest( \
            *(reversed(v.shape) for v in vs), fillvalue=0)))))
    r = np.zeros(shape, dtype=np.result_type(*vs))
    for v in vs:
        r[...,*map(slice, v.shape)] += v
    return r

def vecnpsub(v, w):
    """Return the difference of two vectors."""
    v, w = np.asarray(v), np.asarray(w)
    shape = tuple(reversed(tuple(map(max, zip_longest( \
            reversed(v.shape), reversed(w.shape), fillvalue=0)))))
    r = np.zeros(shape, dtype=np.result_type(v, w))
    r[...,*map(slice, v.shape)] += v
    r[...,*map(slice, w.shape)] -= w
    return r



if __name__ == '__main__':
    from numpy.polynomial.polynomial import polyadd, polysub
    
    
    #creation
    #veczero
    assert np.array_equal(vecnpzero(), [0])
    assert np.array_equal(vecnpzero(1), [[0]])
    assert np.array_equal(vecnpzero(2), [[0],
                                         [0]])
    
    #vecbasis
    assert np.array_equal(vecnpbasis(2, 3), [0, 0, 3])
    assert np.array_equal(vecnpbasis(4, 3, 2), [[0, 0, 0, 0, 3],
                                                [0, 0, 0, 0, 3]])
    
    #vecrand
    assert vecnprand(2).shape == (2,)
    assert vecnprand(2, 3).shape == (3, 2)
    
    #vecgauss
    assert vecnprandn(2).shape == (2,)
    assert vecnprandn(2, d=3).shape == (3, 2)
    assert np.allclose(np.linalg.norm(vecnprandn(2, d=3), axis=1), [1, 1, 1])
    
    
    #veceq
    assert vecnpeq([1, 2], [1, 2, 0])
    assert not vecnpeq([1, 2], [1, 2, 1])
    assert np.array_equal(vecnpeq([[1, 2],
                                   [1, 3]], [1, 2, 0]), [True, False])
    assert np.array_equal(vecnpeq([1, 2], [[1, 2, 0],
                                           [1, 3, 1]]), [True, False])
    assert np.array_equal(vecnpeq([[1, 2],
                                   [3, 4]], [[1, 2, 0],
                                             [3, 4, 1]]), [True, False])
    
    #vectrim
    assert np.array_equal(vecnptrim([0, 0]), [0])
    assert np.array_equal(vecnptrim([0, 5, 1e-10]), [0, 5])
    assert np.array_equal(vecnptrim([[0, 0],
                                     [0, 0]]), [[0],
                                                [0]])
    assert np.array_equal(vecnptrim([[1, 2, 3e-10, 4e-12],
                                     [5, 6, 7    , 8e-13]]), [[1, 2, 3e-10],
                                                              [5, 6, 7]])
    
    #Hilbert
    #vecabsq
    assert vecnpabsq([1, 2]) == 5
    assert np.array_equal(vecnpabsq([[1, 2],
                                     [3, 4]]), [5, 25])
    
    #vecabs
    assert vecnpabs([3, 4]) == 5
    assert np.array_equal(vecnpabs([[3,  4],
                                    [5, 12]]), [5, 13])
    
    #vecdot
    assert vecnpdot([1, 2], [3, 4, 5]) == 11
    assert np.array_equal(vecnpdot([[1, 2],
                                    [3, 4]], [5, 6, 7]), [17, 39])
    assert np.array_equal(vecnpdot([1, 2], [[3, 4, 5],
                                            [6, 7, 8]]), [11, 20])
    assert np.array_equal(vecnpdot([[1, 2],
                                    [3, 4]], [[5, 6,  7],
                                              [8, 9, 10]]), [17, 60])
    
    #vector space
    #vecadd
    assert vecnpadd() == np.array([0])
    assert np.array_equal(vecnpadd([1, 2], [3, 4, 5]), [4, 6, 5])
    assert np.array_equal(vecnpadd([[1, 2],
                                    [3, 4]], [5, 6, 7]), [[6,  8, 7],
                                                          [8, 10, 7]])
    assert np.array_equal(vecnpadd([[1, 2],
                                    [3, 4]], [[5, 6,  7],
                                              [8, 9, 10]]), [[ 6,  8,  7],
                                                             [11, 13, 10]])
    for _ in range(100):
        #1D + 1D
        v = np.random.normal(size=np.random.randint(1, 20))
        w = np.random.normal(size=np.random.randint(1, 20))
        assert np.allclose(vecnpadd(v, w), polyadd(v, w))
        #1D + 2D
        w = np.random.normal(size=
                (np.random.randint(1, 20), np.random.randint(1, 20)))
        assert np.allclose(vecnpadd(v, w), [polyadd(v, wi) for wi in w])
        #2D + 2D
        v = np.random.normal(size=
                (np.random.randint(1, 20), np.random.randint(1, 20)))
        w = np.random.normal(size=(v.shape[0], np.random.randint(1, 20)))
        assert np.allclose(vecnpadd(v, w),
                [polyadd(vi, wi) for vi, wi in zip(v, w)])
    
    #vecsub
    assert np.array_equal(vecnpsub([1, 2], [3, 5, 7]), [-2, -3, -7])
    assert np.array_equal(vecnpsub([[1, 2],
                                    [3, 4]], [5, 7, 9]), [[-4, -5, -9],
                                                          [-2, -3, -9]])
    assert np.array_equal(vecnpsub([[1, 2],
                                    [3, 4]], [[5, 7,  9],
                                              [6, 8, 10]]), [[-4, -5,  -9],
                                                             [-3, -4, -10]])
    for _ in range(100):
        #1D + 1D
        v = np.random.normal(size=np.random.randint(1, 20))
        w = np.random.normal(size=np.random.randint(1, 20))
        assert np.allclose(vecnpsub(v, w), polysub(v, w))
        #1D + 2D
        w = np.random.normal(size=
                (np.random.randint(1, 20), np.random.randint(1, 20)))
        assert np.allclose(vecnpsub(v, w), [polysub(v, wi) for wi in w])
        #2D + 2D
        v = np.random.normal(size=
                (np.random.randint(1, 20), np.random.randint(1, 20)))
        w = np.random.normal(size=(v.shape[0], np.random.randint(1, 20)))
        assert np.allclose(vecnpsub(v, w),
                [polysub(vi, wi) for vi, wi in zip(v, w)])
