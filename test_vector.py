from vector import Vector
from math import isclose, sqrt



assert Vector(2) == Vector((0, 0, 1))
assert isclose(abs(Vector.random(10)), 1)
assert not isclose(abs(Vector.random(10, normed=False)), 1)
assert abs(Vector.ZERO) == 0

v = Vector((1, 2, 3, 4, 5))

assert len(v) == 5
assert v[2] == 3
assert v[999] == 0
assert Vector((1, 2, 3)) == Vector((1, 2, 3, 0))
assert not Vector((1, 2, 3)) == Vector((1, 2, 3, 1))
assert v<<1 == Vector((2, 3, 4, 5))
assert v>>1 == Vector((0, 1, 2, 3, 4, 5))
assert Vector((1, 0)).trim() == Vector((1,)) \
        and Vector(tuple()).trim() == Vector(tuple())

assert isclose(abs(v), sqrt(55))
assert v+Vector((3, 2, 1)) == Vector((4, 4, 4, 4, 5))
assert v+2 == Vector((3, 4, 5, 6, 7))
assert v-Vector((3, 2, 1)) == Vector((-2, 0, 2, 4, 5))
assert 2*v == v*2 == Vector((2, 4, 6, 8, 10))
assert v*Vector((2, 3, 4)) == Vector((2, 6, 12))
assert v/2 == Vector((0.5, 1.0, 1.5, 2, 2.5))
