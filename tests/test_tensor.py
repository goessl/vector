from vector import *
import numpy as np



def test_tenbasis():
    assert np.array_equal(tenbasis((2, 3), c=5), np.array([[0, 0, 0, 0],
                                                           [0, 0, 0, 0],
                                                           [0, 0, 0, 5]]))

def test_tenrank():
    assert tenrank([[[]]]) == 3

def test_tendim():
    assert tendim([[0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 5]]) == (3, 4)

def test_tenpos():
    p = np.random.rand(2, 3)
    assert np.array_equal(p, tenpos(p))

def test_tenneg():
    p = np.random.rand(2, 3)
    assert np.array_equal(-p, tenneg(p))

def test_tenaddc():
    assert np.array_equal(tenaddc(tenzero, 5, (1, 2, 3)), tenbasis((1, 2, 3), 5))
    assert np.array_equal(tenaddc([[1, 2, 3],
                                   [4, 5, 6]], 5, (0, 1, 2)), [[[1, 0, 0],
                                                                [2, 0, 5],
                                                                [3, 0, 0]],
                                                               [[4, 0, 0],
                                                                [5, 0, 0],
                                                                [6, 0, 0]]])

def test_tenadd():
    assert tenadd() ==tenzero
    assert np.array_equal(tenadd([[1, 2, 3],
                                  [4, 5, 6]]), [[1, 2, 3],
                                                [4, 5, 6]])
    assert np.array_equal(tenadd([[1, 2, 3],
                                  [4, 5, 6]], tenzero), [[1, 2, 3],
                                                         [4, 5, 6]])
    assert np.array_equal(tenadd(
            [[1, 2, 3],
             [4, 5, 6]], [[[1, 2, 3, 4],
                           [4, 5, 6, 7]]]), [[[2, 2, 3, 4],
                                              [6, 5, 6, 7],
                                              [3, 0, 0, 0]],
                                             [[4, 0, 0, 0],
                                              [5, 0, 0, 0],
                                              [6, 0, 0, 0]]])

def test_tensub():
    assert np.array_equal(tensub(
            [[1, 2, 3],
             [4, 5, 6]], [[[1, 2, 3, 4],
                           [4, 5, 6, 7]]]), [[[ 0, -2, -3, -4],
                                              [-2, -5, -6, -7],
                                              [ 3,  0,  0,  0]],
                                             [[ 4,  0,  0,  0],
                                              [ 5,  0,  0,  0],
                                              [ 6,  0,  0,  0]]])
    assert np.array_equal(tensub([[1, 2, 3],
                                  [4, 5, 6]], tenzero), [[1, 2, 3],
                                                         [4, 5, 6]])
    assert np.array_equal(tensub(tenzero, [[1, 2, 3],
                                           [4, 5, 6]]), [[-1, -2, -3],
                                                         [-4, -5, -6]])
