"""Vector operations as lazy generators.

```python
>>> from vector import veclbasis
>>> veclbasis(3)
<generator object veclbasis at 0x0123456789ABCDEF>
>>> tuple(veclbasis(3))
(0, 0, 0, 1)
```

Prefixed by `vecl...` (vector - lazy).

Functions are **generators**.

Lazy generator versions of [`functional`](functional.md).

Different behaviour:

- [`vecrandn`][vector.functional.vecrandn]: normalisation not possible.

Not implemented lazily as these are consumers:

- [`vecabs`][vector.functional.vecabs]
- [`vecabsq`][vector.functional.vecabsq]
- [`vecdot`][vector.functional.vecdot]
- [`vecparallel`][vector.functional.vecparallel]
"""



from .creation import *
from .utility import *
from .hilbert_space import *
from .vector_space import *
from .elementwise import *
