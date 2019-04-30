from warnings import warn

import numpy as np

from .ols import Weighted, LinearRegression


def huber(u):
    """u is scaled residual"""
    w = np.zeros_like(u)
    w[np.abs(u) <= 1.345] = 1
    w[np.abs(u) > 1.345] = 1.345 / (np.abs(u[np.abs(u) > 1.345]))
    return w

def dsquare(u):
    """u is scaled residual"""
    w = np.zeros_like(u)
    u_bool = np.abs(u) <= 4.685
    w[u_bool] = (1 - (u[u_bool]/4.685)**2) ** 2
    return w

def scaled_residuel(resd):
    """convert all residual to scaled residual"""
    MAD = np.median(np.abs(resd - np.median(resd))) / 0.6745
    u = resd / MAD
    return u


class IRLS(Weighted):
    def __init__(self, intercept=True):
        self._isintercept = intercept
    
    def fit(self, X, y, method, itermax=100):
        if method not in ['huber', 'dsquare']:
            raise ValueError("")
            
        n, p = X.shape
        self._n = n
        self._p = p
        
        # calculate coefficient and weight
        b, weight = self._calculate(X, y, method, itermax)
        self.coefficient_ = b
        self.weight_ = weight.ravel()
        
        # set intercept column
        if self._isintercept:
            ones = np.ones((X.shape[0], 1))
            X = np.hstack((ones, X))
            self.X_ = X
            self.y_ = y
        else:
            self.X_ = X
            self.y_ = y
        
        # set weight matrix
        W = np.zeros((X.shape[0], X.shape[0]))
        np.fill_diagonal(W, weight)
        
        # calculate some statistics and set as attributes
        ymean = np.average(y.ravel(), weights=weight.ravel())
        yhat = X @ b
        SST = ( (y - ymean).T @ W @ (y - ymean) )[0,0]    # [0,0]: convert from array to scaler
        SSE = ( (y - yhat).T @ W @ (y - yhat) )[0,0]
        SSR = SST - SSE
        self.SS_ = {'SST': SST, 'SSR': SSR, 'SSE': SSE} 
        self.MS_ = {'MSR': SSR/p, 'MSE': SSE/(n-p-1)}
        self.F_ = self.MS_['MSR'] / self.MS_['MSE']
        self.R2_ = SSR / SST
        self.R2_adj_ = 1 - ((n-1) / (n-p-1)) * (SSE / SST)
        self.residual_ = y - yhat
        
        sb = self.MS_['MSE'] * np.linalg.inv(X.T @ W @ X)
        self.coefficient_standard_error_ = np.sqrt(sb.diagonal())[:, np.newaxis]
        self.t_ = b / self.coefficient_standard_error_
        
    def _calculate(self, X, y, method, itermax):
        # do the first estimate
        if method == 'huber':
            weight_func = huber
            lr = LinearRegression(self._isintercept)
            lr.fit(X, y)
            ini_resid = lr.residual_
            weight = weight_func(scaled_residuel(ini_resid))
        elif method == 'dsquare':
            weight_func = dsquare
            ir = IRLS(self._isintercept)
            ir.fit(X, y,  'huber')
            ini_resid = ir.residual_
            weight = weight_func(scaled_residuel(ini_resid))
        
        new_weight = weight
        
        # iterate weights to find coefficient
        for i in range(itermax):
            weight = new_weight
            wr = Weighted(self._isintercept)
            wr.fit(X, y, weight)
            resid = wr.residual_
        
            # check if weight is converge
            new_weight = weight_func(scaled_residuel(resid))
            if np.sum(np.abs(new_weight - weight)) / np.sum(np.abs(weight)) <= 1e-6:
                break

        else:
            # if the iteration times is reach itermax, raise warning
            warn_messg = "Iteration times has reached the 'itermax'." \
                         + " The coefficients may not converge yet."
            warn(warn_messg)
        
        return wr.coefficient_, weight


class LAR:
    pass


class LMS:
    pass