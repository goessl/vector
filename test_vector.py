from vector import *
from math import isclose, sqrt
from random import random



#vecbasis
assert vecbasis(2, c=5) == (0, 0, 5)

#vecrandom
vecrandom(2)

#vecgauss
assert isclose(vecabsq(vecgauss(10)), 1)
assert not isclose(vecabsq(vecgauss(10, normed=False)), 1)


#veceq
assert veceq((), ())
assert veceq((0,), ())
assert veceq((0, 0), (0,))
assert veceq((1,), (1, 0))
assert not veceq((1,), ())

#vectrim
assert vectrim(()) == ()
assert vectrim((0,)) == ()
assert vectrim((1, 0)) == (1,)

#vecround
assert vecround(()) == ()
assert vecround((1.1,)) == (1,)


#vecabssq
assert vecabsq(()) == 0
assert vecabsq((1j, 2, 3j)) == 14

#vecabs
assert vecabs(()) == 0
assert vecabs((1,)) == 1
assert vecabs((3, 4)) == 5

#vecdot
assert vecdot((), ()) == 0
assert vecdot((1,), ()) == 0
assert vecdot((1, 2), (3, 4)) == 11


#vecadd
assert vecadd() == ()
assert vecadd((1, 2)) == (1, 2)
assert vecadd((1, 2, 3), (4, 5)) == (5, 7, 3)

#vecsub
assert vecsub((), ()) == ()
assert vecsub((1,), ()) == (1,)
assert vecsub((), (1,)) == (-1,)
assert vecsub((1, 2, 3), (4, 5)) == (-3, -3, 3)

#vectruediv
assert vectruediv((), 1) == ()
assert vectruediv((4,), 2) == (2,)

#vecfloordiv
assert vecfloordiv((), 1) == ()
assert vecfloordiv((3,), 2) == (1,)



#Vector
assert Vector(2) == Vector((0, 0, 1))
Vector.random(2)
assert isclose(abs(Vector.gauss(10)), 1)
assert not isclose(abs(Vector.gauss(10, normed=False)), 1)
assert abs(Vector.ZERO) == 0

v = Vector((1, 2, 3, 4, 5))

assert len(v) == 5
assert v[2] == 3
assert v[999] == 0
assert v[1:4:2] == Vector((2, 4))
assert v[::].coef == (1, 2, 3, 4, 5)
assert v[:6:].coef == (1, 2, 3, 4, 5, 0)
assert Vector((1, 2, 3)) == Vector((1, 2, 3, 0))
assert not Vector((1, 2, 3)) == Vector((1, 2, 3, 1))
assert v<<1 == Vector((2, 3, 4, 5))
assert v>>1 == Vector((0, 1, 2, 3, 4, 5))
assert Vector((1, 0)).trim() == Vector((1,)) \
        and Vector(tuple()).trim() == Vector(tuple())

assert isclose(abs(v), sqrt(55))
assert v+Vector((3, 2, 1)) == Vector((4, 4, 4, 4, 5))
assert v-Vector((3, 2, 1)) == Vector((-2, 0, 2, 4, 5))
assert 2*v == v*2 == Vector((2, 4, 6, 8, 10))
assert v/2 == Vector((0.5, 1.0, 1.5, 2, 2.5))
assert v//2 == Vector((0, 1, 1, 2, 2))
