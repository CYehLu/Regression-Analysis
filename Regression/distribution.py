import numpy as np
from scipy.special import gamma
from scipy.optimize import root
from scipy import integrate


def gaussian(x, mean, var):
    return np.exp(-(x-mean)**2/(2*var)) / np.sqrt(2*np.pi*var)

def t(x, df):
    return gamma((df+1)/2) / (gamma(df/2)*np.sqrt(np.pi*df)) * (1+x**2/df)**(-(df+1)/2)

def F(x, df1, df2):
    # check x is a numpy ndarray or not
    if type(x).__module__ == 'numpy':
        xn = x.copy()
        xn[xn <= 0] = 0
    else:
        xn = x
    return gamma((df1+df2)/2) * (df1/df2)**(df1/2) * xn**(df1/2-1)   \
           / (gamma(df1/2) * gamma(df2/2) * (1+df1*xn/df2)**(df1/2+df2/2))

def critical_value(func_type, alpha, **kwargs):
    """
    find the critical value that the right area is equal to alpha
    
    Parameter:
    ---------
    func_type: 'F', 't' or 'gaussian'.
    alpha: float
    
    **kwargs:
      dfs: tuple. degree of freedom
      mean_var: tuple. mean and variance for gaussian
      
    Return:
    ------
    a critical value that the right area is equal to alpha
    """
    if kwargs.keys() - set(['dfs', 'mean_var']):
        # kwargs contains some not allowed keyword parameter
        raise ValueError("unrecognized kwargs. only 'dfs' and 'mean_var' available.")
    if not kwargs.get('dfs') and not kwargs.get('mean_var'):
        raise ValueError("miss keyword argument 'dfs' or 'mean_var'.")
    
    dfs = kwargs.get('dfs')
    mean_var = kwargs.get('mean_var')
    
    if func_type == 'F':
        f = lambda x: integrate.quad(F, x, 40, args=dfs)[0] - alpha
    elif func_type == 't':
        f = lambda x: integrate.quad(t, x, 20, args=dfs)[0] - alpha
    elif func_type == 'gaussian':
        f = lambda x: integrate.quad(gaussian, x, mean_var[1]*6, args=mean_var)[0] - alpha

    return root(f, x0=1).x

def find_p_value(test_type, tail, test_statistic, dfs=None):
    """
    find p-value in the right/left/two tail hypothesis test
    
    Parameter:
    ---------
    test_type: str, 'F', 't' or 'z'
    tail: str, 'right', 'left' or 'two'
    test_statistic: scaler, test statistic in the hypothesis test
    dfs: tuple, degree of freedom. it can be ignored if func_type = 'z'
    
    Return:
    ------
    p-value
    """
    if tail == 'right':
        m = 1
        ep = [60, 20, 10]   # endpoint of intergrate
    elif tail == 'left':
        m = -1
        ep = [0, -20, -10]
    elif tail == 'two':
        m = 2
        ep = [60, 20, 10]
        test_statistic = np.abs(test_statistic)
    
    if test_type == 'F':
        return m * integrate.quad(F, test_statistic, test_statistic+ep[0], args=dfs)[0]
    elif test_type == 't':
        return m * integrate.quad(t, test_statistic, test_statistic+ep[1], args=dfs)[0]
    elif test_type == 'z':
        return m * integrate.quad(gaussian, test_statistic, test_statistic+ep[2], args=(0,1))[0]
    else:
        raise ValueError("Unpermitted func_type")