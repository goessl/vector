from vector import *
import numpy as np
from math import isclose, sqrt
from random import random



def test_vecbasis():
    assert vecbasis(2, c=5) == (0, 0, 5)

def test_vecrand():
    v = vecrand(2)
    assert isinstance(v, tuple) and len(v)==2

def test_vecrandn():
    assert isclose(vecabsq(vecrandn(10)), 1)
    assert not isclose(vecabsq(vecrandn(10, normed=False)), 1)


def test_veceq():
    assert veceq((), ())
    assert veceq((0,), ())
    assert veceq((0, 0), (0,))
    assert veceq((1,), (1, 0))
    assert not veceq((1,), ())

def test_vectrim():
    assert vectrim(()) == ()
    assert vectrim((0,)) == ()
    assert vectrim((1, 0)) == (1,)

def test_vecround():
    assert vecround(()) == ()
    assert vecround((1.1,)) == (1,)


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
    assert vecdot((1, 2), (3, 4)) == 11


def test_vecadd():
    assert vecadd() == ()
    assert vecadd((1, 2)) == (1, 2)
    assert vecadd((1, 2, 3), (4, 5)) == (5, 7, 3)

def test_vecsub():
    assert vecsub((), ()) == ()
    assert vecsub((1,), ()) == (1,)
    assert vecsub((), (1,)) == (-1,)
    assert vecsub((1, 2, 3), (4, 5)) == (-3, -3, 3)

def test_vectruediv():
    assert vectruediv((), 1) == ()
    assert vectruediv((4,), 2) == (2,)

def test_vecfloordiv():
    assert vecfloordiv((), 1) == ()
    assert vecfloordiv((3,), 2) == (1,)
