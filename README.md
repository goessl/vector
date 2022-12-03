# Hermite Function Series

A Hermite function series module.

## Operations

In the following let
$$
    f=\sum_{k=0}^\infty f_kh_k, \ g=\sum_{k=0}^\infty g_kh_k.
$$

### Hilbert space operations

#### Scalar product

$$
    <f|g>_{L_\mathbb{R}^2}=\sum_{k=0}^\infty f_k^*g_k
$$

#### Norm

$$
    ||f||_{L_\mathbb{R}^2}=\sqrt{\sum_{k=0}^\infty |f_k|^2}
$$

#### Addition

$$
    f+g=\sum_{k=0}^\infty (f_k+g_k)h_k
$$

### Function operations

#### Evaluation

$$
    f(x)=\sum_{k=0}^\infty f_kh_k(x)=e^{-\frac{x^2}{2}}\sum_{k=0}^\infty f_k\frac{1}{\sqrt{2^kk!\sqrt{\pi}}}H_k(x)
$$

#### Derivative
