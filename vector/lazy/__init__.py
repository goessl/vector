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

Lazy generator versions of [`dense`](dense.md).

Different behaviour:

- [`vecrandn`][vector.dense.vecrandn]: normalisation not possible.

Not implemented lazily as these are consumers:

- [`vecabs`][vector.dense.vecabs]
- [`vecabsq`][vector.dense.vecabsq]
- [`vecdot`][vector.dense.vecdot]
"""



from .creation import *
from .utility import *
from .hilbertspace import *
from .vectorspace import *
from .elementwise import *
