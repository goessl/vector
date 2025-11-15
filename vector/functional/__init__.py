"""Vector operation functions.

```python
>>> from vector import vecadd
>>> a = (5, 6, 7)
>>> b = [2]
>>> c = range(4)
>>> vecadd(a, b, c)
(7, 7, 9, 3)
```

Prefixed by `vec...` (vector).

All functions accept vectors as **single exhaustible iterables**.

They **return vectors as tuples**.

The functions are **type-independent**. However, the data types used must
*support necessary scalar operations*. For instance, for vector addition,
coefficients must be addable.
"""

from .creation import *
from .utility import *
from .hilbert_space import *
from .vector_space import *
from .elementwise import *
