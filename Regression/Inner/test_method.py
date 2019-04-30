import numpy as np

from .distribution import critical_value, find_p_value
from ..ols import LinearRegression
    
def _Brown_Forsythe(y, residual, method='both', alpha=0.05):
    """
    do the Brown Forsythe test (test const variance).
    if p-value is small or X_BP2 <= critical value, 
    null hypothesis will be rejected (H0: variance is const)
    
    Parameter:
    ---------
    y: 1-d array
    residual: 1-d array
    method: 'both', 'p' or 'region'
    alpha: float
    
    Return:
    ------
    if both:
    return (p-value, tBF statistics, critical value)
    if p:
    return p-value
    if region:
    return (tBF statistics, critical value)
    """
    n = residual.size
    
    ## split to two group
    # rearrange y and residual based on y's sort
    argy = np.argsort(y)    
    sort_y = y[argy]
    sort_res = residual[argy]
    # split y into 2 subarray that these 2 range difference is minimum
    y1, y2 = _find_min_rangediff_of_two_subarray(sort_y)   
    # base on splited y1 and y2 to split residual into 2 group
    length = y1.size
    e1, e2 = sort_res[:length], sort_res[length:]
    n1 = e1.size
    n2 = e2.size
    
    # start to calculate test statistics
    e1_med = np.median(e1)
    e2_med = np.median(e2)
    d1 = np.abs(e1 - e1_med)
    d2 = np.abs(e2 - e2_med)

    d1_deviation = d1 - d1.mean()
    d2_deviation = d2 - d2.mean()
    s = np.sqrt((np.sum(d1_deviation**2) + np.sum(d2_deviation**2)) / (n-2))
    
    tBF = (d1.mean() - d2.mean()) / (s * np.sqrt(1/n1 + 1/n2))
    
    if method == 'p' or 'both':
        pvalue = find_p_value('t', 'two', tBF, dfs=(n-2))
    if method == 'region' or 'both':
        critical = critical_value('t', alpha=alpha/2, dfs=(n-2))
        
    if method == 'p':
        return pvalue
    elif method == 'region':
        return (tBF, critical)
    elif method == 'both':
        return (pvalue, tBF, critical)

def _Breusch_Pagan(X, residual, SSE, method='both', alpha=0.05):
    """
    do the Breusch Pagan test (test const variance).
    if p-value is small or abs(tBF statistics) <= critical value, 
    null hypothesis will be rejected (H0: variance is const)
    
    Parameter:
    ---------
    X: 2-d array
    residual: 1-d array
    SSE: SSE in original regression model
    method: 'both', 'p' or 'region'
    alpha: float
    
    Return:
    ------
    if both:
    return (p-value, X_BP2 statistics, critical value)
    if p:
    return p-value
    if region:
    return (X_BP2 statistics, critical value)
    """ 
    #log_residual2 = np.log(residual ** 2)[:, np.newaxis]   # column vector
    residual2 = (residual ** 2)[:, np.newaxis]  # column vector
    n = residual2.size
    p = X.shape[1]       # number of variables
    
    lr = LinearRegression()
    lr.fit(X, residual2)
    SSR_star = lr.SS_['SSR']
    
    X_BP2 = 0.5*SSR_star / (SSE/n)**2
    
    # do the test
    if method == 'p' or 'both':
        pvalue = find_p_value('chi', 'right', X_BP2, dfs=(p))
    if method == 'region' or 'both':
        critical = critical_value('chi', alpha, dfs=(p))
    
    if method == 'p':
        return pvalue
    elif method == 'region':
        return (X_BP2, critical)
    elif method == 'both':
        return (pvalue, X_BP2, critical)
    
    
    
def _find_min_rangediff_of_two_subarray(arr):
    """
    given a sorted array, 
    it will split it into two subarray which range difference is minimum.
    
    this is used for _Brown_Forsythe
    """
    minimum = arr[-1] - arr[0]
    pos = 1
    for i in range(1, len(arr)-1):
        L = arr[i] - arr[0]
        R = arr[-1] - arr[i+1]
        diff = np.abs(R - L)
        if diff <= minimum:
            pos = i
            minimum = diff
    return (arr[:pos+1], arr[pos+1:])