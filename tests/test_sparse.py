from vector import *
from math import isclose, sqrt
from itertools import islice, count
from random import randint, random
from operationcounter import OperationCounter, count_ops
from collections import Counter



#creation
def test_vecszero():
    assert vecszero == {}

def test_vecsbasis():
    for n in range(10):
        c = random()
        assert vecsbasis(n, c) == {n:c}

def test_vecsbases():
    c = random()
    for i, ei in enumerate(islice(vecsbases(start=5, c=c), 10), start=5):
        assert ei == vecsbasis(i, c)

def test_vecsrand():
    v = vecsrand(10)
    assert isinstance(v, dict)
    assert all(isinstance(vi, float) and 0<=vi<1 for vi in v.values())
    assert len(v) == 10
    assert set(range(10)) == v.keys()

def test_vecsrandn():
    assert isclose(vecsabsq(vecsrandn(10)), 1)
    assert not isclose(vecsabsq(vecsrandn(10, normed=False)), 1)


#utility
def test_vecseq():
    assert vecseq(vecszero, vecszero)
    assert vecseq({0:0}, vecszero)
    assert vecseq({0:0, 20:0}, {5:0})
    assert vecseq({1:2}, {1:2, 3:0})
    assert not vecseq({0:1}, vecszero)
    assert not vecseq({0:1, 1:2, 2:3}, {0:1, 1:2, 2:4})

def test_vecstrim():
    assert vecstrim(vecszero) == vecszero
    assert vecstrim({1:0, 2:0, 5:0}) == vecszero
    assert vecstrim({2:0, 5:1, 10:0}) == {5:1}

def test_vecsrshift():
    assert vecsrshift(vecszero, 2) == vecszero
    assert vecsrshift({0:1, 1:2, 2:3}, 2) == {2:1, 3:2, 4:3}

def test_vecslshift():
    assert vecslshift(vecszero, 2) == vecszero
    assert vecslshift({0:1, 1:2, 2:3, 3:4}, 2) == {0:3, 1:4}


#Hilbert space
def test_vecsconj():
    assert vecsconj({0:1, 1:2, 2:3}) == {0:1, 1:2, 2:3}
    assert vecsconj({0:1, 1:2j, 2:3}) == {0:1, 1:-2j, 2:3}
    assert vecsconj({0:1, 1:4+2j, 2:3.5}) == {0:1, 1:4-2j, 2:3.5}

def test_vecsabs():
    assert vecsabs(vecszero) == 0
    assert vecsabs({0:1}) == 1
    assert vecsabs({0:3, 10:4}) == 5
    
    assert isclose(abs(vecsabs({0:1j, 1:2, 2:3j}, conjugate=True)), sqrt(14))
    assert isclose(vecsabs({0:1, 1:2, 5:3}, weights={0:5, 1:6, 5:7, 6:8}), sqrt(92))
    assert isclose(abs(vecsabs({0:1, 1:2+3j, 5:3+4j}, conjugate=True)), sqrt(39))
    assert isclose(abs(vecsabs({0:1, 1:2+3j, 5:3+4j}, weights={0:5, 1:6, 5:7, 6:8}, conjugate=True)), sqrt(258))

def test_vecsabsq():
    assert vecsabsq(vecszero) == 0
    
    assert vecsabsq({0:1j, 1:2, 5:3j}, conjugate=True) == 14
    assert vecsabsq({0:1, 1:2, 5:3}, weights={0:5, 1:6, 5:7, 6:8}) == 92
    assert vecsabsq({0:1, 1:2+3j, 5:3+4j}, conjugate=True) == 39
    assert vecsabsq({0:1, 1:2+3j, 5:3+4j}, weights={0:5, 1:6, 5:7, 6:8}, conjugate=True) == 258

def test_vecsdot():
    assert vecsdot(vecszero, vecszero) == 0
    assert vecsdot({0:1}, vecszero) == 0
    assert vecsdot({0:1, 5:2}, {0:3, 5:4, 6:5}) == 11

#def test_vecparallel():
#    assert vecparallel((1, 2, 3), (3, 4, 5)) == False
#    assert vecparallel((1, 2, 3), (3, 6, 9)) == True


#vector space
def test_vecspos():
    assert vecspos({0:+1, 1:-2, 2:+3}) == {0:+1, 1:-2, 2:+3}

def test_vecsneg():
    assert vecsneg({0:+1, 1:-2, 2:+3}) == {0:-1, 1:+2, 2:-3}

def test_vecsadd():
    assert vecsadd() == vecszero
    assert vecsadd({0:1, 1:2}) == {0:1, 1:2}
    assert vecsadd({0:1, 1:2, 2:3}, {0:4, 1:5}) == {0:5, 1:7, 2:3}

def test_vecsaddc():
    assert vecsaddc(vecszero, 2, 3) == {3:2}
    assert vecsaddc({0:1, 1:2}, 4, 5) == {0:1, 1:2, 5:4}
    assert vecsaddc({0:1, 1:2, 2:3, 3:4, 4:5}, 5, 2) == {0:1, 1:2, 2:8, 3:4, 4:5}

def test_vecssub():
    assert vecssub(vecszero, vecszero) == vecszero
    assert vecssub({0:1}, vecszero) == {0:1}
    assert vecssub(vecszero, {0:1}) == {0:-1}
    assert vecssub({0:1, 1:2, 2:3}, {0:4, 1:5}) == {0:-3, 1:-3, 2:3}

def test_vecssubc():
    assert vecssubc(vecszero, 2, 3) == {3:-2}
    assert vecssubc({0:1, 1:2}, 4, 5) == {0:1, 1:2, 5:-4}
    assert vecssubc({0:1, 1:2, 2:3, 3:4, 4:5}, 5, 2) == {0:1, 1:2, 2:-2, 3:4, 4:5}

def test_vecsrmul():
    assert vecsrmul(5, vecszero) == vecszero
    assert vecsrmul(5, {0:1, 1:2, 2:3}) == {0:5, 1:10, 2:15}

def test_vecstruediv():
    assert vecstruediv(vecszero, 1) == vecszero
    assert vecstruediv({0:4}, 2) == {0:2}

def test_vecsfloordiv():
    assert vecsfloordiv(vecszero, 1) == vecszero
    assert vecsfloordiv({0:3}, 2) == {0:1}

def test_vecsmod():
    assert vecsmod(vecszero, 2) == vecszero
    assert vecsmod({0:3}, 2) == {0:1}

def test_vecsdivmod():
    v, a = {0:1, 1:2, 2:3}, 2
    assert vecsdivmod(v, a) == (vecsfloordiv(v, a), vecsmod(v, a))


#elementwise
def test_vecshadamard():
    assert vecshadamard() == vecszero
    assert vecshadamard(vecszero) == vecszero
    assert vecshadamard(vecszero, vecszero) == vecszero
    assert vecshadamard({0:1}, vecszero) == vecszero
    assert vecshadamard(vecszero, {0:1}) == vecszero
    assert vecshadamard({0:1, 1:2, 2:3}, {0:4, 1:5}, {0:6, 1:7, 2:8, 3:9}) == {0:24, 1:70}

def test_vecshadamardtruediv():
    assert vecshadamardtruediv(vecszero, vecszero) == vecszero
    assert vecshadamardtruediv(vecszero, {0:1}) == vecszero
    assert vecshadamardtruediv({0:1, 1:2, 2:3}, {0:4, 1:5, 2:6, 3:7}) == {0:1/4, 1:2/5, 2:3/6}

def test_vecshadamardfloordiv():
    assert vecshadamardfloordiv(vecszero, vecszero) == vecszero
    assert vecshadamardfloordiv(vecszero, {0:1}) == vecszero
    assert vecshadamardfloordiv({0:1, 1:5, 2:3}, {0:4, 1:2, 2:6}) == {0:0, 1:2, 2:0}

def test_vecshadamardmod():
    assert vecshadamardmod(vecszero, vecszero) == vecszero
    assert vecshadamardmod(vecszero, {0:1}) == vecszero
    assert vecshadamardmod({0:1, 1:2, 2:3}, {0:4, 1:5, 2:6}) == {0:1, 1:2, 2:3}

def test_vecshadamarddivmod():
    v, w = {0:4, 1:5, 2:6}, {0:1, 1:2, 2:3}
    assert vecshadamarddivmod(v, w) == (vecshadamardfloordiv(v, w), vecshadamardmod(v, w))

def test_vecshadamardmin():
    assert vecshadamardmin() == vecszero
    assert vecshadamardmin(vecszero) == vecszero
    u = {0: 3, 1:6, 2:1}
    v = {0:10, 1:3, 2:7, 3:9, 4:10}
    w = {0: 5, 1:2, 2:3, 3:6, 4: 1, 5:9}
    assert vecshadamardmin(u) == u
    assert vecshadamardmin(u, v, w) == {0:3, 1:2, 2:1, 3:6, 4:1, 5:9}

def test_vecshadamardmax():
    assert vecshadamardmax() == vecszero
    assert vecshadamardmax(vecszero) == vecszero
    u = {0: 3, 1:6, 2:1}
    v = {0:10, 1:3, 2:7, 3:9, 4:10}
    w = {0: 5, 1:2, 2:3, 3:6, 4: 1, 5:9}
    assert vecshadamardmax(u, v, w) == {0:10, 1:6, 2:7, 3:9, 4:10, 5:9}
    assert vecshadamardmax(u) == u
