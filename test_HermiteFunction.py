import numpy as np
from HermiteFunction import HermiteFunction


if __name__ == '__main__':
    x = np.linspace(-4, +4, 1000)
    
    
    #def __init__(self, coef):
    #def random(deg, normed=False):
    
    """fit"""
    #try replicating other Hermite function
    for _ in range(100):
        f = HermiteFunction.random(20)
        y = f(x)
        fit = HermiteFunction.fit(x, y, 20)
        assert np.allclose(f.coef, fit.coef)
    #try hitting some points
    #x = np.linspace(-4, +4, 4)
    #for _ in range(100):
    #    y = np.random.uniform(-1, +1, len(x))
    #    fit = HermiteFunction.fit(x, y, 20)
    #    assert np.allclose(y, fit(x))
    
    #Hilbert space stuff
    #def dot(self, other):
    #def __abs__(self):
    
    """mul"""
    for _ in range(100):
        f = HermiteFunction.random(np.random.randint(0, 20))
        c = np.random.rand()
        assert np.allclose((c*f)(x), c*(f(x)))
    
    #def __rmul__(self, other):
    
    """add"""
    for _ in range(100):
        f = HermiteFunction.random(np.random.randint(0, 20))
        g = HermiteFunction.random(np.random.randint(0, 20))
        assert np.allclose((f+g)(x), f(x)+g(x))
    
    #def __radd__(self, other):
    
    #function stuff
    #def __call__(self, x):
    
    """der"""
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
    
    """kin"""
    def kin_num(x, y):
        """Nummeric kinetic energy."""
        x, y_lapl = der_num(x, y, 2)
        y = (y[2:] + 2*y[1:-1] + y[:-2]) / 4 #mid y twice to broadcast to y_lapl
        return -np.trapz(y*y_lapl, x) / 2
    
    for _ in range(100):
        f = HermiteFunction.random(5)
        assert np.isclose(f.kin, kin_num(x, f(x)), atol=1e-2)
        
    #python stuff
    #def __str__(self):
