import pytest
from vector import *
from math import isclose, sqrt
from itertools import islice, count
from random import randint
from operationcounter import OperationCounter, count_ops
from collections import Counter



#creation
def test_veczero():
    assert veczero == ()
    assert veczero(list) == []

def test_vecbasis():
    assert vecbasis(2, c=5) == (0, 0, 5)

def test_vecbases():
    for i, ei in enumerate(islice(vecbases(), 10)):
        assert ei == vecbasis(i)

def test_vecrand():
    v = vecrand(10)
    assert isinstance(v, tuple)
    assert all(isinstance(vi, float) and 0<=vi<1 for vi in v)
    assert len(v) == 10

def test_vecrandn():
    assert isclose(vecabsq(vecrandn(10)), 1)
    assert not isclose(vecabsq(vecrandn(10, normed=False)), 1)



#utility
def test_veclen():
    assert veclen(veczero) == 0
    assert veclen((1, 2, 3)) == 3


def test_veceq():
    assert veceq(veczero, veczero)
    assert veceq((0,), veczero)
    assert veceq((0, 0), (0,))
    assert veceq((1,), (1, 0))
    assert not veceq((1,), veczero)
    assert not veceq((1, 2, 3), (1, 2, 4))


def test_vectrim():
    assert vectrim(veczero) == veczero
    assert vectrim((0,)) == veczero
    assert vectrim((1, 0)) == (1,)

def test_vecitrim():
    v = []
    assert vecitrim(v)==v and v==[]
    v = [0]
    assert vecitrim(v)==v and v==[]
    v = [1, 0]
    assert vecitrim(v)==v and v==[1]


def test_vecrshift():
    assert vecrshift(veczero, 2) == (0, 0)
    assert vecrshift((1, 2, 3), 2) == (0, 0, 1, 2, 3)

def test_vecirshift():
    v = []
    assert vecirshift(v, 2)==v and v==[0, 0]
    v = [1, 2, 3]
    assert vecirshift(v, 2)==v and v==[0, 0, 1, 2, 3]


def test_veclshift():
    assert veclshift(veczero, 2) == veczero
    assert veclshift((1, 2, 3, 4), 2) == (3, 4)

def test_vecilshift():
    v = []
    assert vecilshift(v, 2)==v and v==[]
    v = [1, 2, 3, 4]
    assert vecilshift(v, 2)==v and v==[3, 4]



#Hilbert space
def test_vecconj():
    assert vecconj((1, 2, 3)) == (1, 2, 3)
    assert vecconj((1, 2j, 3)) == (1, -2j, 3)
    assert vecconj((1, 4+2j, 3.5)) == (1, 4-2j, 3.5)

def test_veciconj():
    v = [1, 2, 3]
    assert veciconj(v)==v and v==[1, 2, 3]
    v = [1, 2j, 3]
    assert veciconj(v)==v and v==[1, -2j, 3]
    v = [1, 4+2j, 3.5]
    assert veciconj(v)==v and v==[1, 4-2j, 3.5]


def test_vecabs():
    assert vecabs(veczero) == 0
    assert vecabs((1,)) == 1
    assert vecabs((3, 4)) == 5
    
    assert isclose(abs(vecabs((1j, 2, 3j), conjugate=True)), sqrt(14))
    assert isclose(vecabs((1, 2, 3), weights=(5, 6, 7, 8)), sqrt(92))
    assert isclose(abs(vecabs((1, 2+3j, 3+4j), conjugate=True)), sqrt(39))
    assert isclose(abs(vecabs((1, 2+3j, 3+4j), weights=(5, 6, 7, 8), conjugate=True)), sqrt(258))


def test_vecabsq():
    assert vecabsq(veczero) == 0
    
    assert vecabsq((1j, 2, 3j), conjugate=True) == 14
    assert vecabsq((1, 2, 3), weights=(5, 6, 7, 8)) == 92
    assert vecabsq((1, 2+3j, 3+4j), conjugate=True) == 39
    assert vecabsq((1, 2+3j, 3+4j), weights=(5, 6, 7, 8), conjugate=True) == 258


def test_vecdot():
    assert vecdot(veczero, veczero) == 0
    assert vecdot((1,), veczero) == 0
    assert vecdot((1, 2), (3, 4, 5)) == 11



#vector space
def test_vecpos():
    assert vecpos((+1, -2, +3)) == (+1, -2, +3)

def test_vecipos():
    v = [+1, -2, +3]
    assert vecipos(v)==v and  v==[+1, -2, +3]


def test_vecneg():
    assert vecneg((+1, -2, +3)) == (-1, +2, -3)

def test_vecineg():
    v = [+1, -2, +3]
    assert vecineg(v)==v and  v==[-1, +2, -3]


def test_vecadd():
    assert vecadd() == veczero
    assert vecadd((1, 2)) == (1, 2)
    assert vecadd((1, 2, 3), (4, 5)) == (5, 7, 3)

def test_veciadd():
    v = []
    assert veciadd(v)==v and v==[]
    v = [1, 2]
    assert veciadd(v)==v and v==[1, 2]
    v = [1, 2]
    assert veciadd(v, (3, 4, 5))==v and v==[4, 6, 5]
    v = [1, 2, 3]
    assert veciadd(v, (4, 5))==v and v==[5, 7, 3]


def test_vecaddc():
    assert vecaddc(veczero, 2, 3) == (0, 0, 0, 2)
    assert vecaddc((1, 2), 4, 5) == (1, 2, 0, 0, 0, 4)
    assert vecaddc((1, 2, 3, 4, 5), 5, 2) == (1, 2, 8, 4, 5)


def test_vecsub():
    assert vecsub(veczero, veczero) == veczero
    assert vecsub((1,), veczero) == (1,)
    assert vecsub(veczero, (1,)) == (-1,)
    assert vecsub((1, 2, 3), (4, 5)) == (-3, -3, 3)

def test_vecisub():
    v = [1, 2]
    assert vecisub(v, (3, 4, 5))==v and v==[-2, -2, -5]
    v = [1, 2, 3]
    assert vecisub(v, (4, 5))==v and v==[-3, -3, 3]


def test_vecsubc():
    assert vecsubc(veczero, 2, 3) == (0, 0, 0, -2)
    assert vecsubc((1, 2), 4, 5) == (1, 2, 0, 0, 0, -4)
    assert vecsubc((1, 2, 3, 4, 5), 5, 2) == (1, 2, -2, 4, 5)


def test_vecmul():
    assert vecmul(veczero, 5) == veczero
    assert vecmul((1, 2, 3), 5) == (5, 10, 15)

def test_vecrmul():
    assert vecrmul(5, veczero) == veczero
    assert vecrmul(5, (1, 2, 3)) == (5, 10, 15)

def test_vecimul():
    v = []
    assert vecimul(v, 5)==v and v==[]
    v = [1, 2, 3]
    assert vecimul(v, 5)==v and v==[5, 10, 15]


def test_vectruediv():
    assert vectruediv(veczero, 1) == veczero
    assert vectruediv((1, 2, 3, 4), 2) == (1/2, 2/2, 3/2, 4/2)

def test_vecitruediv():
    v = []
    assert vecitruediv(v, 1)==v and v==[]
    v = [1, 2, 3, 4]
    assert vecitruediv(v, 2)==v and v==[1/2, 2/2, 3/2, 4/2]


def test_vecfloordiv():
    assert vecfloordiv(veczero, 1) == veczero
    assert vecfloordiv((1, 2, 3, 4), 2) == (0, 1, 1, 2)

def test_vecifloordiv():
    v = []
    assert vecifloordiv(v, 1)==v and v==[]
    v = [1, 2, 3, 4]
    assert vecifloordiv(v, 2)==v and v==[0, 1, 1, 2]


def test_vecmod():
    assert vecmod(veczero, 1) == veczero
    assert vecmod((1, 2, 3, 4), 2) == (1, 0, 1, 0)

def test_vecimod():
    v = []
    assert vecimod(v, 1)==v and v==[]
    v = [1, 2, 3, 4]
    assert vecimod(v, 2)==v and v==[1, 0, 1, 0]


def test_vecdivmod():
    v, a = (1, 2, 3), 2
    assert vecdivmod(v, a) == (vecfloordiv(v, a), vecmod(v, a))



#elementwise
def test_vechadamard():
    assert vechadamard() == veczero
    assert vechadamard(veczero) == veczero
    assert vechadamard(veczero, veczero) == veczero
    assert vechadamard((1,), veczero) == veczero
    assert vechadamard(veczero, (1,)) == veczero
    assert vechadamard((1, 2, 3), (4, 5), (6, 7, 8, 9)) == (24, 70)


def test_vechadamardtruediv():
    assert vechadamardtruediv(veczero, veczero) == veczero
    assert vechadamardtruediv(veczero, (1,)) == veczero
    assert vechadamardtruediv((1, 2, 3), (4, 5, 6, 7)) == (1/4, 2/5, 3/6)
    with pytest.raises(ZeroDivisionError):
        vechadamardtruediv((1, 2, 3), (4, 5))

def test_vecihadamardtruediv():
    v = []
    assert vecihadamardtruediv(v, veczero)==v and v==[]
    v = []
    assert vecihadamardtruediv(v, (1,))==v and v==[]
    v = [1, 2, 3]
    assert vecihadamardtruediv(v, (4, 5, 6, 7))==v and v==[1/4, 2/5, 3/6]
    with pytest.raises(ZeroDivisionError):
        v = [1, 2, 3]
        vecihadamardtruediv(v, (4, 5))


def test_vechadamardfloordiv():
    assert vechadamardfloordiv(veczero, veczero) == veczero
    assert vechadamardfloordiv(veczero, (1,)) == veczero
    assert vechadamardfloordiv((1, 5, 3), (4, 2, 6)) == (0, 2, 0)
    with pytest.raises(ZeroDivisionError):
        vechadamardfloordiv((1, 2, 3), (4, 5))

def test_vecihadamardfloordiv():
    v = []
    assert vecihadamardfloordiv(v, veczero)==v and v==[]
    v = []
    assert vecihadamardfloordiv(v, (1,))==v and v==[]
    v = [1, 5, 3]
    assert vecihadamardfloordiv(v, (4, 2, 6))==v and v==[0, 2, 0]
    with pytest.raises(ZeroDivisionError):
        v = [1, 2, 3]
        vecihadamardfloordiv(v, (4, 5))


def test_vechadamardmod():
    assert vechadamardmod(veczero, veczero) == veczero
    assert vechadamardmod(veczero, (1,)) == veczero
    assert vechadamardmod((1, 2, 3), (4, 5, 6)) == (1, 2, 3)
    with pytest.raises(ZeroDivisionError):
        vechadamardmod((1, 2, 3), (4, 5))

def test_vecihadamardmod():
    v = []
    assert vecihadamardmod(v, veczero)==v and v==[]
    v = []
    assert vecihadamardmod(v, (1,))==v and v==[]
    v = [1, 2, 3]
    assert vecihadamardmod(v, (4, 5, 6))==v and v==[1, 2, 3]
    with pytest.raises(ZeroDivisionError):
        v = [1, 2, 3]
        vecihadamardmod(v, (4, 5))


def test_vechadamarddivmod():
    v, w = (4, 5, 6), (1, 2, 3)
    assert vechadamarddivmod(v, w) == (vechadamardfloordiv(v, w), vechadamardmod(v, w))


def test_vechadamardmin():
    assert vechadamardmin() == veczero
    assert vechadamardmin(veczero) == veczero
    u = ( 3, 6, 1)
    v = (10, 3, 7, 9, 10)
    w = ( 5, 2, 3, 6,  1, 9)
    assert vechadamardmin(u) == u
    assert vechadamardmin(u, v, w) == (3, 2, 1, 6, 1, 9)

def test_vechadamardmax():
    assert vechadamardmax() == veczero
    assert vechadamardmax(veczero) == veczero
    u = ( 3, 6, 1)
    v = (10, 3, 7, 9, 10)
    w = ( 5, 2, 3, 6,  1, 9)
    assert vechadamardmax(u, v, w) == (10, 6, 7, 9, 10, 9)
    assert vechadamardmax(u) == u
