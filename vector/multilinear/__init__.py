"""Tensor arithmetic.

Prefixed by `ten...` (tensor).

Handle multiaxis vectors, that for example represent multivariate polynomials.

Tensors are returned as `numpy.ndarray`s.

Broadcasting happens similar to [`numpy`s broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html),
but the axes are matched in ascending order instead of descending order, and
the arrays don't get stretched but rather padded with zeros.
"""



from .creation import *
from .utility import *
from .hilbert_space import *
from .vector_space import *
from .elementwise import *
