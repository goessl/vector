from vector import *
from math import isclose
from itertools import islice, count
from random import randint
from operationcounter import OperationCounter, count_ops
from collections import Counter



#creation
#veczero
#vecbasis
#vecbases
#vecrand
#vecrandn

#utility
def test_veceq():
    for _ in range(20):
        n, m = randint(0, 20), randint(0, 20)
        v = tuple(OperationCounter(randint(-100, +100)) for _ in range(n))
        w = tuple(OperationCounter(randint(-100, +100)) for _ in range(m))
        with count_ops() as counter:
            veceq(v, w)
        assert counter <= Counter({'eq':min(n, m), 'bool':abs(n-m)})

#vectrim
#vecround
#vecrshift
#veclshift

#Hilbert space
#vecconj

def test_vecabs():
    for n in range(20):
        v = tuple(OperationCounter(randint(-100, +100)) for _ in range(n))
        with count_ops() as counter:
            vecabs(v)
        assert counter.keys() <= {'mul', 'add', 'pow'}
        assert counter['mul'] == n
        assert counter['pow'] == (1 if n>=1 else 0)
        if n >= 1:
            assert counter['add'] == n-1
        if n <= 1:
            assert counter['add'] == 0
        
        with count_ops() as counter:
            vecabs(v, weights=count())
        assert counter.keys() <= {'mul', 'add', 'pow'}
        assert counter['mul'] == 2*n
        assert counter['pow'] == (1 if n>=1 else 0)
        if n >= 1:
            assert counter['add'] == n-1
        if n <= 1:
            assert counter['add'] == 0

def test_vecabsq():
    for n in range(20):
        v = tuple(OperationCounter(randint(-100, +100)) for _ in range(n))
        with count_ops() as counter:
            vecabsq(v)
        if n >= 1:
            assert counter == Counter({'mul':n, 'add':n-1})
        if n <= 1:
            assert counter == Counter({'mul':n})
        
        with count_ops() as counter:
            vecabsq(v, weights=count())
        if n >= 1:
            assert counter == Counter({'mul':2*n, 'add':n-1})
        if n <= 1:
            assert counter == Counter({'mul':2*n})

def test_vecdot():
    for _ in range(20):
        n, m = randint(0, 20), randint(0, 20)
        v = tuple(OperationCounter(randint(-100, +100)) for _ in range(n))
        w = tuple(OperationCounter(randint(-100, +100)) for _ in range(m))
        with count_ops() as counter:
            vecdot(v, w)
        if min(n, m) >= 1:
            assert counter == Counter({'mul':min(n, m), 'add':min(n, m)-1})
        if min(n, m) <= 1:
            assert counter == Counter({'mul':min(n, m)})

def test_vecparallel():
    assert vecparallel((1, 2, 3), (3, 4, 5)) == False
    assert vecparallel((1, 2, 3), (3, 6, 9)) == True


#vector space
def test_vecpos():
    for n in range(20):
        v = tuple(OperationCounter(randint(-100, +100)) for _ in range(n))
        with count_ops() as counter:
            vecpos(v)
        assert counter == Counter({'pos':n})

def test_vecneg():
    for n in range(20):
        v = tuple(OperationCounter(randint(-100, +100)) for _ in range(n))
        with count_ops() as counter:
            vecneg(v)
        assert counter == Counter({'neg':n})

def test_vecadd():
    for _ in range(20):
        n, m = randint(0, 20), randint(0, 20)
        v = tuple(OperationCounter(randint(-100, +100)) for _ in range(n))
        w = tuple(OperationCounter(randint(-100, +100)) for _ in range(m))
        with count_ops() as counter:
            vecadd(v, w)
        assert counter == Counter({'add': min(n, m)})

def test_vecaddc():
    for _ in range(20):
        v = tuple(OperationCounter(randint(-100, +100)) for _ in range(randint(0, 20)))
        c, i = OperationCounter(randint(-100, +100)), randint(0, 20)
        with count_ops() as counter:
            vecaddc(v, c, i)
        assert counter==Counter({'add': 1}) or counter==Counter({'pos': 1})

def test_vecsub():
    for _ in range(20):
        n, m = randint(0, 20), randint(0, 20)
        v = tuple(OperationCounter(randint(-100, +100)) for _ in range(n))
        w = tuple(OperationCounter(randint(-100, +100)) for _ in range(m))
        with count_ops() as counter:
            vecsub(v, w)
        assert counter == Counter({'sub':min(m, n), 'neg':(m-n if m>=n else 0)})

def test_vecsubc():
    for _ in range(20):
        v = tuple(OperationCounter(randint(-100, +100)) for _ in range(randint(0, 20)))
        c, i = OperationCounter(randint(-100, +100)), randint(0, 20)
        with count_ops() as counter:
            vecsubc(v, c, i)
        assert counter==Counter({'sub': 1}) or counter==Counter({'neg': 1})

def test_vecmul():
    for n in range(20):
        a = randint(-100, +100)
        v = tuple(OperationCounter(randint(-100, +100)) for _ in range(n))
        with count_ops() as counter:
            vecmul(a, v)
        assert counter == Counter({'rmul':n})

def test_vectruediv():
    for n in range(20):
        a = randint(-100, +99) or +100
        v = tuple(OperationCounter(randint(-100, +100)) for _ in range(n))
        with count_ops() as counter:
            vectruediv(v, a)
        assert counter == Counter({'truediv':n})

def test_vecfloordiv():
    for n in range(20):
        a = randint(-100, +99) or +100
        v = tuple(OperationCounter(randint(-100, +100)) for _ in range(n))
        with count_ops() as counter:
            vecfloordiv(v, a)
        assert counter == Counter({'floordiv':n})

def test_vecmod():
    for n in range(20):
        a = randint(-100, +99) or +100
        v = tuple(OperationCounter(randint(-100, +100)) for _ in range(n))
        with count_ops() as counter:
            vecmod(v, a)
        assert counter == Counter({'mod':n})

def test_vecdivmod():
    for n in range(20):
        a = randint(-100, +99) or +100
        v = tuple(OperationCounter(randint(-100, +100)) for _ in range(n))
        with count_ops() as counter:
            vecdivmod(v, a)
        assert counter == Counter({'divmod':n})


#elementwise
def test_vechadamard():
    for _ in range(20):
        N = randint(0, 10)
        ns = [randint(0, 20) for _ in range(N)]
        vs = [tuple(OperationCounter(randint(-100, +100)) for _ in range(n)) for n in ns]
        with count_ops() as counter:
            vechadamard(*vs)
        if N>=2 and min(ns)>=1:
            assert counter == Counter({'mul': (N-1)*min(ns)})
        else:
            assert counter == Counter({'mul': 0})

def test_vechadamardtruediv():
    for n in range(20):
        v = tuple(OperationCounter(randint(-100, +100)) for _ in range(n))
        w = tuple(OperationCounter(randint(-100, +99) or +100) for _ in range(randint(n, 100)))
        with count_ops() as counter:
            vechadamardtruediv(v, w)
        assert counter == Counter({'truediv':n})

def test_vechadamardfloordiv():
    for n in range(20):
        v = tuple(OperationCounter(randint(-100, +100)) for _ in range(n))
        w = tuple(OperationCounter(randint(-100, +99) or +100) for _ in range(randint(n, 100)))
        with count_ops() as counter:
            vechadamardfloordiv(v, w)
        assert counter == Counter({'floordiv':n})

def test_vechadamardmod():
    for n in range(20):
        v = tuple(OperationCounter(randint(-100, +100)) for _ in range(n))
        w = tuple(OperationCounter(randint(-100, +99) or +100) for _ in range(randint(n, 100)))
        with count_ops() as counter:
            vechadamardmod(v, w)
        assert counter == Counter({'mod':n})

def test_vechadamarddivmod():
    for n in range(20):
        v = tuple(OperationCounter(randint(-100, +100)) for _ in range(n))
        w = tuple(OperationCounter(randint(-100, +99) or +100) for _ in range(randint(n, 100)))
        with count_ops() as counter:
            vechadamarddivmod(v, w)
        assert counter == Counter({'divmod':n})

def test_vechadamardmin():
    for _ in range(20):
        n, m = randint(0, 20), randint(0, 20)
        v = tuple(OperationCounter(randint(-100, +100)) for _ in range(n))
        w = tuple(OperationCounter(randint(-100, +100)) for _ in range(m))
        with count_ops() as counter:
            vechadamardmin(v, w)
        assert counter == Counter({'lt':min(n, m)})

def test_vechadamardmax():
    for _ in range(20):
        n, m = randint(0, 20), randint(0, 20)
        v = tuple(OperationCounter(randint(-100, +100)) for _ in range(n))
        w = tuple(OperationCounter(randint(-100, +100)) for _ in range(m))
        with count_ops() as counter:
            vechadamardmax(v, w)
        assert counter == Counter({'gt':min(n, m)})
