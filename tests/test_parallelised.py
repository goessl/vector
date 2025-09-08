from vector import *
import numpy as np
from numpy.polynomial.polynomial import polyadd, polysub



#creation
def test_vecnpzero():
    assert np.array_equal(vecnpzero(), [0])
    assert np.array_equal(vecnpzero(1), [[0]])
    assert np.array_equal(vecnpzero(2), [[0],
                                         [0]])

def test_vecnpbasis():
    assert np.array_equal(vecnpbasis(2, 3), [0, 0, 3])
    assert np.array_equal(vecnpbasis(4, 3, 2), [[0, 0, 0, 0, 3],
                                                [0, 0, 0, 0, 3]])

def test_vecnprand():
    assert vecnprand(2).shape == (2,)
    assert vecnprand(2, 3).shape == (3, 2)

def test_vecnprandn():
    assert vecnprandn(2).shape == (2,)
    assert vecnprandn(2, d=3).shape == (3, 2)
    assert np.allclose(np.linalg.norm(vecnprandn(2, d=3), axis=1), [1, 1, 1])


#utility
def test_vecnpeq():
    assert vecnpeq([1, 2], [1, 2, 0])
    assert not vecnpeq([1, 2], [1, 2, 1])
    assert np.array_equal(vecnpeq([[1, 2],
                                   [1, 3]], [1, 2, 0]), [True, False])
    assert np.array_equal(vecnpeq([1, 2], [[1, 2, 0],
                                           [1, 3, 1]]), [True, False])
    assert np.array_equal(vecnpeq([[1, 2],
                                   [3, 4]], [[1, 2, 0],
                                             [3, 4, 1]]), [True, False])

def test_vecnptrim():
    assert np.array_equal(vecnptrim([0, 0]), [0])
    assert np.array_equal(vecnptrim([0, 5, 1e-10]), [0, 5])
    assert np.array_equal(vecnptrim([[0, 0],
                                     [0, 0]]), [[0],
                                                [0]])
    assert np.array_equal(vecnptrim([[1, 2, 3e-10, 4e-12],
                                     [5, 6, 7    , 8e-13]]), [[1, 2, 3e-10],
                                                              [5, 6, 7]])

def test_vecnpround():
    assert np.array_equal(vecnpround([1.1, 2.2]), [1, 2])
    assert np.array_equal(vecnpround([1.12, 2.23], 1), [1.1, 2.2])


#Hilbert space
def test_vecnpabsq():
    assert vecnpabsq([1, 2]) == 5
    assert np.array_equal(vecnpabsq([[1, 2],
                                     [3, 4]]), [5, 25])

def test_vecnpabs():
    assert vecnpabs([3, 4]) == 5
    assert np.array_equal(vecnpabs([[3,  4],
                                    [5, 12]]), [5, 13])

def test_vecnpdot():
    assert vecnpdot([1, 2], [3, 4, 5]) == 11
    assert np.array_equal(vecnpdot([[1, 2],
                                    [3, 4]], [5, 6, 7]), [17, 39])
    assert np.array_equal(vecnpdot([1, 2], [[3, 4, 5],
                                            [6, 7, 8]]), [11, 20])
    assert np.array_equal(vecnpdot([[1, 2],
                                    [3, 4]], [[5, 6,  7],
                                              [8, 9, 10]]), [17, 60])

def test_vecnpparallel():
    assert vecnpparallel([1, 2, 3], [3, 4, 5]) == False
    assert vecnpparallel([1, 2, 3], [3, 6, 9]) == True
    assert np.array_equal(vecnpparallel([1, 2, 3], [[3, 4, 5],
                                                    [3, 6, 9]]), [False, True])
    assert np.array_equal(vecnpparallel([[1, 2, 3],
                                         [4, 5, 6]], [[3, 4, 5],
                                                      [8, 10, 12]]),
                                         [False, True])


#vector space
def test_vecnppos():
    assert np.array_equal(vecnppos([+1, -2, +3]), [+1, -2, +3])

def test_vecnpneg():
    assert np.array_equal(vecnpneg([+1, -2, +3]), [-1, +2, -3])

def test_vecnpadd():
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

def test_vecnpsub():
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

def test_vecnpmul():
    assert np.array_equal(vecnpmul(5, [1, 2, 3]), [5, 10, 15])

def test_vecnptruediv():
    assert np.array_equal(vecnptruediv([[1, 10, 16],
                                        [6, 22, 39]], 5), [[1/5, 10/5, 16/5],
                                                           [6/5, 22/5, 39/5]])

def test_vecnpfloordiv():
    assert np.array_equal(vecnpfloordiv([[1, 10, 16],
                                         [6, 22, 39]], 5), [[0, 2, 3],
                                                            [1, 4, 7]])

def test_vecnpmod():
    assert np.array_equal(vecnpmod([[1, 10, 16],
                                    [6, 22, 39]], 5), [[1, 0, 1],
                                                       [1, 2, 4]])
