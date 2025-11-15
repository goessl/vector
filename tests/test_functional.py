from vector import *
from math import isclose, sqrt
from itertools import islice, count
from random import randint
from operationcounter import OperationCounter, count_ops
from collections import Counter



#creation
def test_veczero():
    assert veczero == veczero

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

def test_vecround():
    assert vecround(veczero) == veczero
    assert vecround((1.1, 2.2)) == (1, 2)
    assert vecround((1.12, 2.23), ndigits=1) == (1.1, 2.2)

def test_vecrshift():
    assert vecrshift(veczero, 2) == (0, 0)
    assert vecrshift((1, 2, 3), 2) == (0, 0, 1, 2, 3)

def test_veclshift():
    assert veclshift(veczero, 2) == veczero
    assert veclshift((1, 2, 3, 4), 2) == (3, 4)


#Hilbert space
def test_vecconj():
    assert vecconj((1, 2, 3)) == (1, 2, 3)
    assert vecconj((1, 2j, 3)) == (1, -2j, 3)
    assert vecconj((1, 4+2j, 3.5)) == (1, 4-2j, 3.5)

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

def test_vecparallel():
    assert vecparallel((1, 2, 3), (3, 4, 5)) == False
    assert vecparallel((1, 2, 3), (3, 6, 9)) == True


#vector space
def test_vecpos():
    assert vecpos((+1, -2, +3)) == (+1, -2, +3)

def test_vecneg():
    assert vecneg((+1, -2, +3)) == (-1, +2, -3)

def test_vecadd():
    assert vecadd() == veczero
    assert vecadd((1, 2)) == (1, 2)
    assert vecadd((1, 2, 3), (4, 5)) == (5, 7, 3)

def test_vecaddc():
    assert vecaddc(veczero, 2, 3) == (0, 0, 0, 2)
    assert vecaddc((1, 2), 4, 5) == (1, 2, 0, 0, 0, 4)
    assert vecaddc((1, 2, 3, 4, 5), 5, 2) == (1, 2, 8, 4, 5)

def test_vecsub():
    assert vecsub(veczero, veczero) == veczero
    assert vecsub((1,), veczero) == (1,)
    assert vecsub(veczero, (1,)) == (-1,)
    assert vecsub((1, 2, 3), (4, 5)) == (-3, -3, 3)

def test_vecsubc():
    assert vecsubc(veczero, 2, 3) == (0, 0, 0, -2)
    assert vecsubc((1, 2), 4, 5) == (1, 2, 0, 0, 0, -4)
    assert vecsubc((1, 2, 3, 4, 5), 5, 2) == (1, 2, -2, 4, 5)

def test_vecmul():
    assert vecmul(5, veczero) == veczero
    assert vecmul(5, (1, 2, 3)) == (5, 10, 15)

def test_vectruediv():
    assert vectruediv(veczero, 1) == veczero
    assert vectruediv((4,), 2) == (2,)

def test_vecfloordiv():
    assert vecfloordiv(veczero, 1) == veczero
    assert vecfloordiv((3,), 2) == (1,)

def test_vecmod():
    assert vecmod(veczero, 2) == veczero
    assert vecmod((3,), 2) == (1,)

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

def test_vechadamardfloordiv():
    assert vechadamardfloordiv(veczero, veczero) == veczero
    assert vechadamardfloordiv(veczero, (1,)) == veczero
    assert vechadamardfloordiv((1, 5, 3), (4, 2, 6)) == (0, 2, 0)

def test_vechadamardmod():
    assert vechadamardmod(veczero, veczero) == veczero
    assert vechadamardmod(veczero, (1,)) == veczero
    assert vechadamardmod((1, 2, 3), (4, 5, 6)) == (1, 2, 3)

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
