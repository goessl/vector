from iteration import MISSING, prod_default
from typing import Any, Callable
from collections.abc import Mapping



__all__ = ('vecshadamard', 'vecshadamardtruediv',
           'vecshadamardfloordiv', 'vecshadamardmod', 'vecshadamarddivmod',
           'vecshadamardmin', 'vecshadamardmax')



def vecshadamard(*vs:Mapping[int,Any]) -> dict[int,Any]:
    r"""Return the elementwise product.
    
    $$
        \left((\vec{v}_0)_i\cdot(\vec{v}_1)_i\cdot\cdots\right)_i
    $$
    """
    r = {}
    if not vs:
        return r
    for k in set(vs[0].keys()).intersection(*(v.keys() for v in vs[1:])):
        r[k] = prod_default((v[k] for v in vs), initial=MISSING, default=MISSING)
    return r

def vecshadamardtruediv(v:Mapping[int,Any], w:Mapping[int,Any]) -> dict[int,Any]:
    r"""Return the elementwise true quotient.
    
    $$
        \left(\frac{v_i}{w_i}\right)_i
    $$
    """
    r = {}
    for i, vi in v.items():
        try:
            wi = w[i]
        except KeyError:
            raise ZeroDivisionError
        r[i] = vi / wi
    return r

def vecshadamardfloordiv(v:Mapping[int,Any], w:Mapping[int,Any]) -> dict[int,Any]:
    r"""Return the elementwise floor quotient.
    
    $$
        \left(\left\lfloor\frac{v_i}{w_i}\right\rfloor\right)_i
    $$
    """
    r = {}
    for i, vi in v.items():
        try:
            wi = w[i]
        except KeyError:
            raise ZeroDivisionError
        r[i] = vi // wi
    return r

def vecshadamardmod(v:Mapping[int,Any], w:Mapping[int,Any]) -> dict[int,Any]:
    r"""Return the elementwise remainder.
    
    $$
        \left(v_i \bmod w_i\right)_i
    $$
    """
    r = {}
    for i, vi in v.items():
        try:
            wi = w[i]
        except KeyError:
            raise ZeroDivisionError
        r[i] = vi % wi
    return r

def vecshadamarddivmod(v:Mapping[int,Any], w:Mapping[int,Any]) -> dict[int,Any]:
    r"""Return the elementwise floor quotient and remainder.
    
    $$
        \left(\left\lfloor\frac{v_i}{w_i}\right\rfloor\right)_i, \ \left(v_i \bmod w_i\right)_i
    $$
    """
    q, r = {}, {}
    for i, vi in v.items():
        try:
            wi = w[i]
        except KeyError:
            raise ZeroDivisionError
        q[i], r[i] = divmod(vi, wi)
    return q, r

def vecshadamardmin(*vs:Mapping[int,Any], key:Callable[[Any],Any]|None=None) -> dict[int,Any]:
    r"""Return the elementwise minimum.
    
    $$
        \left(\min((\vec{v}_0)_i,(\vec{v}_1)_i,\cdots)\right)_i
    $$
    """
    r = {}
    if not vs:
        return r
    for k in set(vs[0].keys()).union(*(v.keys() for v in vs[1:])):
        r[k] = min(v[k] for v in vs if k in v)
    return r

def vecshadamardmax(*vs:Mapping[int,Any], key:Callable[[Any],Any]|None=None) -> dict[int,Any]:
    r"""Return the elementwise maximum.
    
    $$
        \left(\max((\vec{v}_0)_i,(\vec{v}_1)_i,\cdots)\right)_i
    $$
    """
    r = {}
    if not vs:
        return r
    for k in set(vs[0].keys()).union(*(v.keys() for v in vs[1:])):
        r[k] = max(v[k] for v in vs if k in v)
    return r
