import numpy as np
from numpy.polynomial.hermite import Hermite
from math import factorial
from scipy.special import binom
from itertools import zip_longest



class HermiteFunction:
    """A Hermite function series class."""
    
    #construction stuff
    def __init__(self, coef):
        """Creates a new Hermite function series.
        
        Parameters
        ----------
        coef : int or array_like
            Index of hermite function
            or list/array of coefficients in order of increasing degree.
        
        Returns
        -------
        hermite function series : HermiteFunction
            A new Hermite function series object.
        """
        if isinstance(coef, int):
            coef = [0]*coef + [1]
        self.coef = np.array(coef)
    
    def random(deg, normed=False):
        """Creates a Hermite function series
        with random coefficients in [-1, 1[.
        
        Parameters
        ----------
        deg : int
            Degree of the series (index deg not included).
        normed : boolean, optional
            True if the coefficients should be normed
            to euclidian length 1 (default: False).
        
        Returns
        -------
        hermite function series : HermiteFunction
            A new Hermite function series object.
        """
        coef = np.random.uniform(-1, +1, deg)
        if normed:
            coef /= np.linalg.norm(coef)
        return HermiteFunction(coef)
    
    def fit(x, y, deg):
        """Creates a least squares Hermite series fit.
        
        Parameters
        ----------
        x : number or array
            x values.
        y : number or array
            y values.
        deg : positive int
            Degree series (index deg not included).
        
        Returns
        -------
        hermite function series : HermiteFunction
            A least squares Hermite series fit.
        """
        #https://de.wikipedia.org/wiki/Multiple_lineare_Regression
        X_T = np.array([HermiteFunction(i)(x) for i in range(deg)])
        X = X_T.T
        X_T_X = X_T @ X
        X_T_X_inv = np.linalg.inv(X_T_X)
        coef = np.dot(X_T_X_inv @ X_T, y)
        
        return HermiteFunction(coef)
    
    
    #Hilbert space stuff
    def dot(self, other):
        """Returns the $L_\mathbb{R}^2$ dot product of self with other.
        
        Parameters
        ----------
        other : HermiteFunction
            Other factor.
        
        Returns
        -------
        dot product : complex
            The dot product of self and other.
        """
        return np.vdot(np.pad(self.coef,
                    (0, max(0, len(other.coef)-len(self.coef)))),
                np.pad(other.coef,
                    (0, max(0, len(self.coef)-len(other.coef)))))
    
    def norm(self):
        """Returns $L_\mathbb{R}^2$ norm of self.
        
        Returns
        -------
        norm : number
            The norm of self.
        """
        return np.linalg.norm(self.coef)
    
    def __mul__(self, other):
        """Scalar/elementwise multiplies other to the coefficients
        and returns the result as a new series object.
        
        Parameters
        ----------
        other : number, array or HermiteFunction
            Other factor.
        
        Returns
        -------
        hermite function series : HermiteFunction
            The product of self and other.
        """
        if isinstance(other, HermiteFunction):
            return HermiteFunction([a*b for a, b \
                in zip_longest(self.coef, other.coef, fillvalue=0)])
        else:
            return HermiteFunction(self.coef * other)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __add__(self, other):
        """Adds other to the coefficients
        and returns the result as a new series object.
        
        Parameters
        ----------
        other : number, array or HermiteFunction
            Other summand.
        
        Returns
        -------
        hermite function series : HermiteFunction
            The sum of self and other.
        """
        if isinstance(other, HermiteFunction):
            return HermiteFunction([a+b for a, b \
                in zip_longest(self.coef, other.coef, fillvalue=0)])
        else:
            return HermiteFunction(self.coef + other)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    
    #function stuff
    def __call__(self, x):
        """Evaluates the series at the given point(s).
        
        Parameters
        ----------
        x : number or array
            Point(s) where to evaluate the series.
        
        Returns
        -------
        y : number or array
            The value(s) at the given point(s).
        """
        y = 0
        for n, c in enumerate(self.coef):
            y += Hermite([0]*n+[c])(x) \
                / np.sqrt(2**n * factorial(n) * np.sqrt(np.pi))
        return y * np.exp(-x**2 / 2)
    
    def der(self, n=1):
        """Returns the n-th derivative of this series.
        
        Parameters
        ----------
        n : positive integer, optional
            Degree of differentiation (default: 1).
        
        Returns
        -------
        hermite function series : HermiteFunction
            The n-th derivative of this series.
        """
        coef = self.coef
        for _ in range(n):
            i = np.arange(len(coef)+1)
            coef = np.append(coef[1:], [0, 0])*np.sqrt((i+1)/2) \
                    - np.append([0], coef)*np.sqrt(i/2)
        return HermiteFunction(coef)
    
    def prod_reorder(self, other):
        """Returns the product of self and other, divided by h_0.
        
        Parameters
        ----------
        other : HermiteFunction
            Other factor.
        
        Returns
        -------
        hermite function series : HermiteFunction
            The product of self and other, divided by h_0.
        """
        coef = np.zeros(len(self.coef)+len(other.coef)-1)
        for b in range(len(coef)):
            for n in range(b, len(self.coef)+len(other.coef)-1, 2):
                for d in range(-b, b+1, 2):
                    i, j = (n-d)//2, (n+d)//2
                    k = (n-b)//2
                    if 0<=i<len(self.coef) and 0<=j<len(other.coef):
                        coef[b] += self.coef[i] * other.coef[j] \
                            * factorial(k) * binom(i, k) * binom(j, k) \
                            * np.sqrt(factorial(b)/(factorial(i)*factorial(j)))
        return HermiteFunction(coef)
    
    def kin(self):
        """Returns the kinetic energy of this series.
        
        Returns
        -------
        kinetic energy : float
            The kinetic energy of this series.
        """
        return self.der().norm()**2 / 2
    
    
    #python stuff
    def __str__(self):
        """Returns a Latex representation.
        
        Returns
        -------
        string : string
            A Latex representation.
        """
        s = f'{self.coef[0]:.1f} h_0'
        for i, c in enumerate(self.coef[1:]):
            s += f' + {c:.1f} h_{i+1}'
        return s
