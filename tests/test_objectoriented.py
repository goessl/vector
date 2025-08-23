from vector import *
from math import isclose, sqrt



def test_init():
    assert Vector(2) == Vector((0, 0, 1))

def test_rand():
    v = Vector.rand(2)
    assert len(v)==2

def test_vecrandn():
    assert isclose(abs(Vector.randn(10)), 1)
    assert not isclose(abs(Vector.randn(10, normed=False)), 1)


v = Vector((1, 2, 3, 4, 5))

def test_len():
    assert len(v) == 5

def test_getitem():
    assert v[2] == 3
    assert v[999] == 0
    assert v[1:4:2] == Vector((2, 4))
    assert v[::].coef == (1, 2, 3, 4, 5)
    assert v[:6:].coef == (1, 2, 3, 4, 5, 0)

def test_veceq():
    assert Vector((1, 2, 3)) == Vector((1, 2, 3, 0))
    assert not Vector((1, 2, 3)) == Vector((1, 2, 3, 1))

def test_shift():
    assert v<<1 == Vector((2, 3, 4, 5))
    assert v>>1 == Vector((0, 1, 2, 3, 4, 5))

def test_vectrim():
    assert Vector((1, 0)).trim() == Vector((1,)) \
            and Vector(tuple()).trim() == Vector(tuple())


def test_vecabsq():
    assert v.absq() == 55

def test_vecabs():
    assert isclose(abs(v), sqrt(55))
    assert abs(Vector.ZERO) == 0


def test_vecadd():
    assert v+Vector((3, 2, 1)) == Vector((4, 4, 4, 4, 5))

def test_vecsub():
    assert v-Vector((3, 2, 1)) == Vector((-2, 0, 2, 4, 5))

def test_vecmul():
    assert 2*v == v*2 == Vector((2, 4, 6, 8, 10))

def test_vectruediv():
    assert v/2 == Vector((0.5, 1.0, 1.5, 2, 2.5))

def test_vecfloordiv():
    assert v//2 == Vector((0, 1, 1, 2, 2))
