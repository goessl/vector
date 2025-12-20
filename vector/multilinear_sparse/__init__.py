"""Sparse tensor arithmetic.

Prefixed by `tens...` (tensor - sparse).

Handle sparse multiaxis tensors, that for example represent multivariate polynomials.

Sparse tensors are accepted and returned as `dict`s where the keys are **trimmed** (no trailing zeros), **non-negative `int` `tuples`**.
"""



from .creation import *
from .conversion import *
from .utility import *
from .hilbert_space import *
from .vector_space import *
from .elementwise import *
