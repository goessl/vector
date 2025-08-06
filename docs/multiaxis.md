# Multiaxis

Prefixed by `ten...` (tensor).

Handle multiaxis vectors, that for example represent multivariate polynomials.

Results are returned as `numpy` arrays.

Broadcasting happens similar to [`numpy`s broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html), but the axes are matched in ascending order instead of descending order, and the arrays don't get stretched but rather padded with zeros.

## creation

- `tenzero`: Zero tensor.

- `tenbasis(i, c=1)`: Return the `i`-th basis tensor times `c`.

- `tenrand(*d)`: Wrapper for `numpy.random.rand`.

- `tenrandn(*d)`: Wrapper for `numpy.random.randn`.

## utility

- `tenrank(t)`: Return the rank of the tensor.

- `tendim(t)`: Return the dimensionalities of the tensor.

- `tentrim(t, tol=1e-9)`: Remove all trailing near zero (abs(v_i)<=tol) coefficients.

- `tenround(t, ndigits=0)`: Wrapper for `numpy.round`.

## vector space

- `tenpos(t)`: Return the tensor with the unary positive operator applied.

- `tenneg(t)`: Return the tensor with the unary negative operator applied.

- `tenaddc(t, c, i=(0,))`: Return `t` with `c` added to the `i`-th coefficient. More efficient than `tenadd(v, tenbasis(i, c)`.

- `tenadd(*ts)`: Return the sum of tensors.

- `tensub(s, t)`: Return the difference of two tensors.

- `tenmul(a, t)`: Return the product of a scalar and a tensor.

- `tentruediv(t, a)`: Return the true division of a tensor and a scalar.

- `tenfloordiv(t, a)`: Return the floor division of a tensor and a scalar.

- `tenmod(t, a)`: Return the elementwise mod of a tensor and a scalar.

## elementwise

- `tenhadamard(*ts)`: Return the elementwise product of tensors.

- `tenhadamardtruediv(s, t)`: Return the elementwise true division of two tensors.

- `tenhadamardfloordiv(s, t)`: Return the elementwise floor division of two tensors.

- `tenhadamardmod(s, t)`: Return the elementwise mod of two tensors.
