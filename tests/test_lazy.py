from vector import *
from itertools import islice



#creation
def test_veclzero():
    assert tuple(veclzero()) == veczero

def test_veclbasis():
    assert tuple(veclbasis(2, c=5)) == vecbasis(2, c=5)

def test_veclbases():
    for i, ei in enumerate(islice(veclbases(), 10)):
        assert tuple(ei) == vecbasis(i)

def test_veclrand():
    v = tuple(veclrand(10))
    assert isinstance(v, tuple)
    assert all(isinstance(vi, float) and 0<=vi<1 for vi in v)
    assert len(v) == 10

def test_veclrandn():
    v = tuple(veclrandn(10))
    assert isinstance(v, tuple)
    assert all(isinstance(vi, float) for vi in v)
    assert len(v) == 10


#utility
def test_vectrim():
    assert tuple(vecltrim(veclzero())) == vectrim(veczero)
    assert tuple(vecltrim((0,))) == vectrim((0,))
    assert tuple(vecltrim((1, 0))) == vectrim((1, 0))

def test_veclround():
    assert tuple(veclround(veclzero())) == vecround(veczero)
    assert tuple(veclround((1.1, 2.2))) == vecround((1.1, 2.2))
    assert tuple(veclround((1.12, 2.23), ndigits=1)) == vecround((1.12, 2.23), ndigits=1)

def test_veclrshift():
    assert tuple(veclrshift(veclzero(), 2)) == vecrshift(veczero, 2)
    assert tuple(veclrshift((1, 2, 3), 2)) == vecrshift((1, 2, 3), 2)

def test_vecllshift():
    assert tuple(vecllshift(veclzero(), 2)) == veclshift(veczero, 2)
    assert tuple(vecllshift((1, 2, 3, 4), 2)) == veclshift((1, 2, 3, 4), 2)


#Hilbert space
def test_veclconj():
    assert tuple(veclconj((1, 2, 3))) == vecconj((1, 2, 3))
    assert tuple(veclconj((1, 2j, 3))) == vecconj((1, 2j, 3))
    assert tuple(veclconj((1, 4+2j, 3.5))) == vecconj((1, 4+2j, 3.5))

#def test_vecabs():
#    assert vecabs(()) == 0
#    assert vecabs((1,)) == 1
#    assert vecabs((3, 4)) == 5

#def test_vecabsq():
#    assert vecabsq(()) == 0
#    assert vecabsq((1j, 2, 3j), conjugate=True) == 14

#def test_vecdot():
#    assert vecdot((), ()) == 0
#    assert vecdot((1,), ()) == 0
#    assert vecdot((1, 2), (3, 4, 5)) == 11

#def test_vecparallel():
#    assert vecparallel((1, 2, 3), (3, 4, 5)) == False
#    assert vecparallel((1, 2, 3), (3, 6, 9)) == True


#vector space
def test_veclpos():
    assert tuple(veclpos((+1, -2, +3))) == vecpos((+1, -2, +3))

def test_veclneg():
    assert tuple(veclneg((+1, -2, +3))) == vecneg((+1, -2, +3))

def test_vecadd():
    assert tuple(vecladd()) == vecadd()
    assert tuple(vecladd((1, 2))) == vecadd((1, 2))
    assert tuple(vecladd((1, 2, 3), (4, 5))) == vecadd((1, 2, 3), (4, 5))

def test_vecaddc():
    assert tuple(vecladdc(veclzero(), 2, 3)) == vecaddc(veczero, 2, 3)
    assert tuple(vecladdc((1, 2), 4, 5)) == vecaddc((1, 2), 4, 5)
    assert tuple(vecladdc((1, 2, 3, 4, 5), 5, 2)) == vecaddc((1, 2, 3, 4, 5), 5, 2)

def test_veclsub():
    assert tuple(veclsub(veclzero(), veclzero())) == vecsub(veczero, veczero)
    assert tuple(veclsub((1,), veclzero())) == vecsub((1,), veczero)
    assert tuple(veclsub(veclzero(), (1,))) == vecsub(veczero, (1,))
    assert tuple(veclsub((1, 2, 3), (4, 5))) == vecsub((1, 2, 3), (4, 5))

def test_veclsubc():
    assert tuple(veclsubc(veclzero(), 2, 3)) == vecsubc(veczero, 2, 3)
    assert tuple(veclsubc((1, 2), 4, 5)) == vecsubc((1, 2), 4, 5)
    assert tuple(veclsubc((1, 2, 3, 4, 5), 5, 2)) == vecsubc((1, 2, 3, 4, 5), 5, 2)

def test_veclmul():
    assert tuple(veclmul(5, veclzero())) == vecmul(5, veczero)
    assert tuple(veclmul(5, (1, 2, 3))) == vecmul(5, (1, 2, 3))

def test_vecltruediv():
    assert tuple(vecltruediv(veclzero(), 1)) == vectruediv((), 1)
    assert tuple(vecltruediv((4,), 2)) == vectruediv((4,), 2)

def test_veclfloordiv():
    assert tuple(veclfloordiv(veclzero(), 1)) == vecfloordiv(veczero, 1)
    assert tuple(veclfloordiv((3,), 2)) == vecfloordiv((3,), 2)

def test_veclmod():
    assert tuple(vecmod(veclzero(), 2)) == vecmod(veczero, 2)
    assert tuple(vecmod((3,), 2)) == vecmod((3,), 2)

def test_vecldivmod():
    v, a = (1, 2, 3), 2
    assert tuple(zip(*vecldivmod(v, a))) == vecdivmod(v, a)


#elementwise
def test_veclhadamard():
    assert tuple(vechadamard()) == vechadamard()
    assert tuple(vechadamard(veclzero())) == vechadamard(veczero)
    assert tuple(vechadamard(veclzero(), veclzero())) == vechadamard(veczero,  veczero)
    assert tuple(vechadamard((1,), veclzero())) == vechadamard((1,), veczero)
    assert tuple(vechadamard(veclzero(), (1,))) == vechadamard(veczero, (1,))
    assert tuple(vechadamard((1, 2, 3), (4, 5), (6, 7, 8, 9))) == vechadamard((1, 2, 3), (4, 5), (6, 7, 8, 9))

def test_veclhadamardtruediv():
    assert tuple(veclhadamardtruediv(veclzero(), veclzero())) == vechadamardtruediv(veczero, veczero)
    assert tuple(veclhadamardtruediv(veclzero(), (1,))) == vechadamardtruediv(veczero, (1,))
    assert tuple(veclhadamardtruediv((1, 2, 3), (4, 5, 6, 7))) == vechadamardtruediv((1, 2, 3), (4, 5, 6, 7))

def test_veclhadamardfloordiv():
    assert tuple(veclhadamardfloordiv(veclzero(), veclzero())) == vechadamardfloordiv(veczero, veczero)
    assert tuple(veclhadamardfloordiv(veclzero(), (1,))) == vechadamardfloordiv(veczero, (1,))
    assert tuple(veclhadamardfloordiv((1, 5, 3), (4, 2, 6))) == vechadamardfloordiv((1, 5, 3), (4, 2, 6))

def test_veclhadamardmod():
    assert tuple(veclhadamardmod(veclzero(), veclzero())) == vechadamardmod(veczero, veczero)
    assert tuple(veclhadamardmod(veclzero(), (1,))) == vechadamardmod(veczero, (1,))
    assert tuple(veclhadamardmod((1, 2, 3), (4, 5, 6))) == vechadamardmod((1, 2, 3), (4, 5, 6))

def test_veclhadamardmin():
    assert tuple(veclhadamardmin()) == vechadamardmin()
    assert tuple(veclhadamardmin(veclzero())) == vechadamardmin(veczero)
    u = ( 3, 6, 1)
    v = (10, 3, 7, 9, 10)
    w = ( 5, 2, 3, 6,  1, 9)
    assert tuple(veclhadamardmin(u)) ==  vechadamardmin(u)
    assert tuple(veclhadamardmin(u, v, w)) == vechadamardmin(u, v, w)

def test_veclhadamardmax():
    assert tuple(veclhadamardmax()) == vechadamardmax()
    assert tuple(veclhadamardmax(veclzero())) == vechadamardmax(veczero)
    u = ( 3, 6, 1)
    v = (10, 3, 7, 9, 10)
    w = ( 5, 2, 3, 6,  1, 9)
    assert tuple(veclhadamardmax(u)) ==  vechadamardmax(u)
    assert tuple(veclhadamardmax(u, v, w)) == vechadamardmax(u, v, w)
