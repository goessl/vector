import numpy as np
from HermiteFunction import HermiteFunction



if __name__ == '__main__':
    x = np.linspace(-4, +4, 1000)
    
    
    
    #HermiteFunction
    #HermiteFunction.random
    
    #fitting
    for _ in range(100):
        deg = np.random.randint(0, 20)
        f = HermiteFunction.random(deg)
        y = f(x)
        fit = HermiteFunction.fit(x, y, deg)
        assert np.allclose(f.coef, fit.coef)
    
    
    
    f = HermiteFunction.random(20)
    #length
    assert len(f) == 21
    #indexing
    f[5]
    assert f[999] == 0
    #iterating
    for c in f:
        pass
    #comparison
    assert f != HermiteFunction(21)
    #shifting
    assert (f<<1).deg == 19 and (f>>1).deg == 21
    
    
    
    #norm
    assert np.isclose(abs(f), 1)
    #dot
    assert np.isclose(f @ HermiteFunction(21), 0)
    
    
    
    #addition & subtraction
    for _ in range(100):
        f = HermiteFunction.random(np.random.randint(0, 20))
        g = HermiteFunction.random(np.random.randint(0, 20))
        assert np.allclose((f+g)(x), f(x)+g(x))
        assert np.allclose((f-g)(x), f(x)-g(x))
    
    #scalar multiplication and division
    for _ in range(100):
        f = HermiteFunction.random(np.random.randint(0, 20))
        c = np.random.rand()
        assert np.allclose((c*f)(x), c*(f(x)))
        assert np.allclose((f/c)(x), (f(x))/c)
    
    
    
    f = HermiteFunction.random(20)
    #degree
    assert f.deg == 20
    #calling
    f(x)
    
    #derivative
    def der_num(x, y, n=1):
        """Nummerical differentiation."""
        for _ in range(n):
            y = np.diff(y) / np.diff(x)
            x = (x[1:] + x[:-1]) / 2
        return x, y
    
    for _ in range(100):
        f = HermiteFunction(np.random.randint(0, 20))
        assert np.allclose(f.der()(der_num(x, f(x))[0]),
                der_num(x, f(x))[1], atol=1e-3)
    
    #kintetic energy
    def kin_num(x, y):
        """Nummeric kinetic energy."""
        x, y_lapl = der_num(x, y, 2)
        y = (y[2:] + 2*y[1:-1] + y[:-2]) / 4 #mid y twice to broadcast to y_lapl
        return -np.trapz(y*y_lapl, x) / 2
    
    for _ in range(100):
        f = HermiteFunction.random(5)
        assert np.isclose(f.kin, kin_num(x, f(x)), atol=1e-2)
    
    
    
    #python stuff
    str(f)
