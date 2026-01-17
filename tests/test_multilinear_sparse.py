from vector import *
import numpy as np



#creation
def test_tenszero():
    assert tenszero == {}

def test_tensbasis():
    assert tensbasis((1, 2, 3, 0), 4) == {(1, 2, 3):4}

def test_tensrand():
    t = tensrand(1, 2, 3)
    assert isinstance(t, dict) and tensrank(t)==3 and tensdim(t)==(1, 2, 3)
    assert all(isinstance(ti, float) and 0<=ti<1 for ti in t.values())

def test_tensrandn():
    t = tensrandn(1, 2, 3)
    assert isinstance(t, dict) and tensrank(t)==3 and tensdim(t)==(1, 2, 3)
    assert all(isinstance(ti, float) for ti in t.values())


#utility
def test_tensrank():
    assert tensrank({(1, 2, 3):5}) == 3

def test_tendim():
    assert tensdim({(1, 2, 3):5}) == (2, 3, 4)

def test_tentrim():
    assert tenstrim({(1, 2, 3):4, (5, 6, 7, 8):0}) == {(1, 2, 3):4}
    assert tenstrim(tenszero) == tenszero


#vector space
def test_tesnpos():
    assert tenspos({(1, 2, 3):+4}) == {(1, 2, 3):+4}

def test_tensneg():
    assert tensneg({(1, 2, 3):+4}) == {(1, 2, 3):-4}

def test_tensadd():
    assert tensadd() == tenszero
    assert tensadd({(1, 2, 3):4}, tenszero) == {(1, 2, 3):4}
    assert tensadd({(1, 2, 3):4}, {(1, 2, 3):5, (4, 5):7}) == {(1, 2, 3):9, (4, 5):7}

def test_tensaddc():
    assert tensaddc({(1, 2, 3):4}, 5) == {():5, (1, 2, 3):4}

def test_tenssub():
    assert tenssub({(1, 2, 3):4}, tenszero) == {(1, 2, 3):4}
    assert tenssub({(1, 2, 3):4}, {(1, 2, 3):5, (4, 5):7}) == {(1, 2, 3):-1, (4, 5):-7}

def test_tenssubc():
    assert tenssubc({(1, 2, 3):4}, 5) == {():-5, (1, 2, 3):4}

def test_tensrmul():
    assert tensrmul(5, tenszero) == tenszero
    assert tensrmul(5, {(1, 2, 3):4}) == {(1, 2, 3):20}

def test_tenstruediv():
    assert tenstruediv(tenszero, 5) == tenszero
    assert tenstruediv({(1, 2, 3):4}, 5) == {(1, 2, 3):4/5}

def test_tensfloordiv():
    assert tensfloordiv(tenszero, 5) == tenszero
    assert tensfloordiv({(1, 2, 3):5}, 4) == {(1, 2, 3):5//4}

def test_tenmod():
    assert tensmod(tenszero, 5) == tenszero
    assert tensmod({(1, 2, 3):5}, 4) == {(1, 2, 3):5%4}


#elementwise
def test_tenshadamard():
    assert tenshadamard() == tenszero
    assert tenshadamard(tenszero) == tenszero
    assert tenshadamard({(1, 2, 3):4}, {(1, 2, 3):4, (4, 5):6}) == {(1, 2, 3):16}

def test_tenshadamardtruediv():
    assert tenshadamardtruediv(tenszero, {(1, 2, 3):4}) == tenszero
    assert tenshadamardtruediv({(1, 2, 3):4}, {(1, 2, 3):5, (4, 5):6}) == {(1, 2, 3):4/5}

def test_tensfloordiv():
    assert tenshadamardfloordiv(tenszero, {(1, 2, 3):4}) == tenszero
    assert tenshadamardfloordiv({(1, 2, 3):4}, {(1, 2, 3):5, (4, 5):6}) == {(1, 2, 3):4//5}

def test_tenshadamardmod():
    assert tenshadamardmod(tenszero, {(1, 2, 3):4}) == tenszero
    assert tenshadamardmod({(1, 2, 3):5}, {(1, 2, 3):4, (4, 5):6}) == {(1, 2, 3):5%4}
