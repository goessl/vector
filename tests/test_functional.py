from vector import *
import numpy as np
from math import isclose, sqrt
from random import random
from itertools import islice



#creation
def test_veczero():
    assert veczero == ()

def test_vecbasis():
    assert vecbasis(2, c=5) == (0, 0, 5)

def test_vecbases():
    for i, vi in enumerate(islice(vecbases(), 10)):
        assert vi == vecbasis(i)

def test_vecrand():
    v = vecrand(2)
    assert isinstance(v, tuple) and len(v)==2

def test_vecrandn():
    assert isclose(vecabsq(vecrandn(10)), 1)
    assert not isclose(vecabsq(vecrandn(10, normed=False)), 1)


#utility
def test_veceq():
    assert veceq((), ())
    assert veceq((0,), ())
    assert veceq((0, 0), (0,))
    assert veceq((1,), (1, 0))
    assert not veceq((1,), ())
    assert not veceq((1, 2, 3), (1, 2, 4))

def test_vectrim():
    assert vectrim(()) == ()
    assert vectrim((0,)) == ()
    assert vectrim((1, 0)) == (1,)

def test_vecround():
    assert vecround(()) == ()
    assert vecround((1.1, 2.2)) == (1, 2)
    assert vecround((1.12, 2.23), ndigits=1) == (1.1, 2.2)

def test_vecrshift():
    assert vecrshift((), 2) == (0, 0)
    assert vecrshift((1, 2, 3), 2) == (0, 0, 1, 2, 3)

def test_veclshift():
    assert veclshift((), 2) == ()
    assert veclshift((1, 2, 3, 4), 2) == (3, 4)


#Hilbert space
def test_vecabsq():
    assert vecabsq(()) == 0
    assert vecabsq((1j, 2, 3j)) == 14

def test_vecabs():
    assert vecabs(()) == 0
    assert vecabs((1,)) == 1
    assert vecabs((3, 4)) == 5

def test_vecdot():
    assert vecdot((), ()) == 0
    assert vecdot((1,), ()) == 0
    assert vecdot((1, 2), (3, 4, 5)) == 11

def test_vecparallel():
    assert vecparallel((1, 2, 3), (3, 4, 5)) == False
    assert vecparallel((1, 2, 3), (3, 6, 9)) == True


#vector space
def test_vecpos():
    assert vecpos((+1, -2, +3)) == (+1, -2, +3)

def test_vecneg():
    assert vecneg((+1, -2, +3)) == (-1, +2, -3)

def test_vecaddc():
    assert vecaddc((), 2, 3) == (0, 0, 0, 2)
    assert vecaddc((1, 2), 4, 5) == (1, 2, 0, 0, 0, 4)
    assert vecaddc((1, 2, 3, 4, 5), 5, 2) == (1, 2, 8, 4, 5)

def test_vecadd():
    assert vecadd() == ()
    assert vecadd((1, 2)) == (1, 2)
    assert vecadd((1, 2, 3), (4, 5)) == (5, 7, 3)

def test_vecsub():
    assert vecsub((), ()) == ()
    assert vecsub((1,), ()) == (1,)
    assert vecsub((), (1,)) == (-1,)
    assert vecsub((1, 2, 3), (4, 5)) == (-3, -3, 3)

def test_vecmul():
    assert vecmul(5, veczero) == veczero
    assert vecmul(5, (1, 2, 3)) == (5, 10, 15)

def test_vectruediv():
    assert vectruediv((), 1) == ()
    assert vectruediv((4,), 2) == (2,)

def test_vecfloordiv():
    assert vecfloordiv((), 1) == ()
    assert vecfloordiv((3,), 2) == (1,)

def test_vecmod():
    assert vecmod((), 2) == ()
    assert vecmod((3,), 2) == (1,)

def test_vecdivmod():
    v, a = (1, 2, 3), 2
    assert vecdivmod(v, a) == (vecfloordiv(v, a), vecmod(v, a))


#elementwise
def test_vechadamard():
    assert vechadamard((), ()) == ()
    assert vechadamard((1,), ()) == ()
    assert vechadamard((), (1,)) == ()
    assert vechadamard((1, 2, 3), (4, 5), (6, 7, 8, 9)) == (24, 70)

def test_vechadamardtruediv():
    assert vechadamardtruediv((), ()) == ()
    assert vechadamardtruediv((), (1,)) == ()
    assert vechadamardtruediv((1, 2, 3), (4, 5, 6, 7)) == (1/4, 2/5, 3/6)

def test_vechadamardfloordiv():
    assert vechadamardfloordiv((), ()) == ()
    assert vechadamardfloordiv((), (1,)) == ()
    assert vechadamardfloordiv((1, 5, 3), (4, 2, 6)) == (0, 2, 0)

def test_vechadamardmod():
    assert vechadamardmod((), ()) == ()
    assert vechadamardmod((), (1,)) == ()
    assert vechadamardmod((1, 2, 3), (4, 5, 6)) == (1, 2, 3)

def test_vechadamardmin():
    u = ( 3, 6, 1)
    v = (10, 3, 7, 9, 10)
    w = ( 5, 2, 3, 6,  1, 9)
    assert vechadamardmin(u, v, w) == (3, 2, 1, 6, 1, 9)
    assert vechadamardmin(u) == u
    assert vechadamardmin(()) == vechadamardmin() == ()

def test_vechadamardmax():
    u = ( 3, 6, 1)
    v = (10, 3, 7, 9, 10)
    w = ( 5, 2, 3, 6,  1, 9)
    assert vechadamardmax(u, v, w) == (10, 6, 7, 9, 10, 9)
    assert vechadamardmax(u) == u
    assert vechadamardmax(()) == vechadamardmax() == ()
