import numpy as np
from numpy.polynomial.hermite import Hermite
from math import factorial
from itertools import zip_longest



class HermiteFunction:
    """A Hermite function series class."""
    
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
    
    #TODO
    def fit(x, y, deg):
        """
        Least squares fit of Hermite series to data.
        """
        #https://gitlab.tugraz.at/397B2A1B074DD115/computational-physics/-/blob/master/ex2/Submission/1_enzyme.py
        #https://de.wikipedia.org/wiki/Multiple_lineare_Regression
        def multiple_linear_regression(X, y):
            X_T = np.transpose(X)
            X_T_X = np.matmul(X_T, X)
            X_T_X_inv = np.linalg.inv(X_T_X)
            return np.dot(np.matmul(X_T_X_inv, X_T), y)
        
        X = np.array([HermiteFunction(i)(x) for i in range(deg)]).transpose()
        coef = multiple_linear_regression(X, y)
        return HermiteFunction(coef)
    
    def random(deg, normed=True):
        """Creates a Hermite function series
        with random coefficients in [-1, 1[.
        
        Parameters
        ----------
        deg : int
            Degree series (index deg not included).
        normed : boolean, optional
            True if the coefficients should be normed to euclidian length 1.
        
        Returns
        -------
        hermite function series : HermiteFunction
            A new Hermite function series object.
        """
        coef = np.random.uniform(-1, +1, deg)
        if normed:
            coef /= np.linalg.norm(coef)
        return HermiteFunction(coef)
    
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
    
    def __add__(self, other):
        """Adds other to the coefficients
        and returns the result as a new series object.
        
        Parameters
        ----------
        other : HermiteFunction, number or array
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
    
    def __rmul__(self, other):
        """Scalar multiplication."""
        return HermiteFunction(self.coef * other)
    
    def __matmul__(self, other):
        """Function multiplication."""
        coef = np.zeros(len(self.coef)+len(other.coef)-1)
        for b in range(len(coef)):
            for n in range(b, len(self.coef)+len(other.coef)-1, 2):
                for d in range(-b, b+1, 2):
                    i, j = (n-d)//2, (n+d)//2
                    k = (n-b)//2
                    if 0<=i<len(self.coef) and 0<=j<len(other.coef):
                        coef[b] += self.coef[i] * other.coef[j] \
                            * factorial(k) * binom(i, k) * binom(j, k) \
                            * sqrt(factorial(b) / (factorial(i)*factorial(j)))
        return HermiteFunction(coef)
    
    def der(self):
        """Returns the derivative of this series.
        
        Returns
        -------
        hermite function series : HermiteFunction
            The derivative of this series.
        """
        i = np.arange(len(self.coef)+1)
        coef = np.append(self.coef[1:], [0, 0])*np.sqrt((i+1)/2) - np.append([0], self.coef)*np.sqrt(i/2)
        return HermiteFunction(coef)
    '''
    def lapl(self):
        """Second derivative."""
        i = np.arange(len(self.coef)+2)
        #special care for h_0 only needed (padding with min)
        coef = np.append(self.coef[2:], [0]*min(len(self.coef)+2,4)) * np.sqrt((i+1)*(i+2))/2 \
            - np.append(self.coef, [0, 0]) * (i+1/2) \
            + np.append([0, 0], self.coef) * np.sqrt((i-1)*i)/2
        return HermiteFunction(coef)
    '''
    def kin(self):
        """Returns the kinetic energy of this series.
        
        Returns
        -------
        kinetic energy : float
            The kinetic energy of this series.
        """
        #return -sum(np.append(self.coef, [0, 0]) * self.lapl().coef) / 2
        return sum(self.der().coef**2) / 2
    
    def __str__(self):
        """Returns a unicode representation.
        
        Returns
        -------
        string : string
            A unicode representation.
        """
        s = f'{self.coef[0]:.1f} h_0'
        for i, c in enumerate(self.coef[1:]):
            s += f' + {c:.1f} h_{i+1}'
        return s
