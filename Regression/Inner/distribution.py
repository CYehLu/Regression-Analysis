import numpy as np
from scipy.special import gamma
from scipy.optimize import root
from scipy import integrate
import scipy.stats


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

def chi_square(x, df):
    return (0.5**(df/2) * x**(df/2-1) * np.exp(-x/2)) / gamma(df/2)

def critical_value(func_type, alpha, **kwargs):
    """
    find the critical value that the right area is equal to alpha
    
    Parameter:
    ---------
    func_type: 'F', 't', 'chi' or 'gaussian'.
    alpha: float
    
    **kwargs:
      dfs: tuple. degree of freedom
      mean_var: tuple. mean and variance for gaussian
      
    Return:
    ------
    a critical value that the right area is equal to alpha
    """
    if func_type == 'F':
        dfn, dfd = kwargs.get('dfs')
        return scipy.stats.f.ppf(1-alpha, dfn=dfn, dfd=dfd)
    elif func_type == 't':
        df = kwargs.get('dfs')
        return scipy.stats.t.ppf(1-alpha, df=df)
    elif func_type == 'chi':
        df = kwargs.get('dfs')
        return scipy.stats.chi2.ppf(1-alpha, df=df)
    elif func_type == 'gaussian':
        mean_var = kwargs.get('mean_var')
        return scipy.stats.norm.ppf(1-alpha, loc=mean_var[0], scale=np.sqrt(mean_var[1]))
    
def find_p_value(test_type, tail, test_statistic, dfs=None):
    """
    find p-value in the right/left/two tail hypothesis test
    
    Parameter:
    ---------
    test_type: str, 'F', 't', 'z' or 'chi'
    tail: str, 'right', 'left' or 'two'
    test_statistic: scaler, test statistic in the hypothesis test
    dfs: tuple, degree of freedom. it can be ignored if func_type = 'z'
    
    Return:
    ------
    p-value
    """
    if test_type == 'F':
        if tail == 'right':
            return 1 - scipy.stats.f.cdf(test_statistic, dfn=dfs[0], dfd=dfs[1])
        elif tail == 'left':
            return scipy.stats.f.cdf(test_statistic, dfn=dfs[0], dfd=dfs[1])

    if test_type == 't':
        if tail == 'right':
            return 1 - scipy.stats.t.cdf(test_statistic, df=dfs)
        elif tail == 'left':
            return scipy.stats.t.cdf(test_statistic, df=dfs)
        elif tail == 'two':
            test_statistic = -np.abs(test_statistic)
            return 2 * scipy.stats.t.cdf(test_statistic, df=dfs)
        
    if test_type == 'chi':
        if tail == 'right':
            return 1 - scipy.stats.chi2.cdf(test_statistic, df=dfs)
        elif tail == 'left':
            return scipy.stats.chi2.cdf(test_statistic, df=dfs)
        
    if test_type == 'z':
        if tail == 'right':
            return 1 - scipy.stats.norm.cdf(test_statistic)
        elif tail == 'left':
            return scipy.stats.norm.cdf(test_statistic)
        elif tail == 'two':
            test_statistic = -np.abs(test_statistic)
            return 2 * scipy.stats.norm.cdf(test_statistic)






'''

def critical_value(func_type, alpha, **kwargs):
    """
    find the critical value that the right area is equal to alpha
    
    Parameter:
    ---------
    func_type: 'F', 't', 'chi' or 'gaussian'.
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
    elif func_type == 'chi':
        f = lambda x: integrate.quad(chi_square, x, 80, args=dfs)[0] - alpha
    elif func_type == 'gaussian':
        f = lambda x: integrate.quad(gaussian, x, mean_var[1]*6, args=mean_var)[0] - alpha

    return root(f, x0=1).x

def find_p_value(test_type, tail, test_statistic, dfs=None):
    """
    find p-value in the right/left/two tail hypothesis test
    
    Parameter:
    ---------
    test_type: str, 'F', 't', 'z' or 'chi'
    tail: str, 'right', 'left' or 'two'
    test_statistic: scaler, test statistic in the hypothesis test
    dfs: tuple, degree of freedom. it can be ignored if func_type = 'z'
    
    Return:
    ------
    p-value
    """
    if tail == 'right':
        m = 1
        ep = [60, 20, 10, 80]   # endpoint of intergrate
    elif tail == 'left':
        m = -1
        ep = [0, -20, -10, 0]
    elif tail == 'two':
        m = 2
        ep = [60, 20, 10, 80]
        test_statistic = np.abs(test_statistic)
    
    if test_type == 'F':
        return m * integrate.quad(F, test_statistic, test_statistic+ep[0], args=dfs)[0]
    elif test_type == 't':
        return m * integrate.quad(t, test_statistic, test_statistic+ep[1], args=dfs)[0]
    elif test_type == 'z':
        return m * integrate.quad(gaussian, test_statistic, test_statistic+ep[2], args=(0,1))[0]
    elif test_type == 'chi':
        return return m * integrate.quad(chi_square, test_statistic, test_statistic+ep[2], args=dfs)[0]
    else:
        raise ValueError("Unpermitted func_type")
        
'''