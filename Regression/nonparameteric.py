import numpy as np

from .ols import LinearRegression, Weighted


class Lowess:
    def __init__(self):
        pass
    
    def fit(self, X, y, xh, q=0.5, isscale=True):
        """
        Fit data to the Lowess model.
        
        Parameter:
        ---------
        X: 2-d array-like.
           The rows of X mean to observation data, and columns mean to variables (feature).
        y: 2-d array-like with shape = (n, 1), where n is the number of observation data and
           is equal to X.shape[0].
        xh: 2-d array-like.
            The values which wanted to predict by lowess. The xh.shape[1] should equal to
            X.shape[1].
        q: float between 0 to 1.
           The ratio of total data number to nearest data number. It is recommended that q
           is between 0.4 to 0.6. Default is 0.5.
        isscale: bool, whether to scale data by deviding std in the distance calculation.
                 Default is True.
                 
        Attribute:
        ---------
        X_: input X.
        y_: input y.
        coefmatrix_: the matrix which rows are the coefficients of regression model to the
                     corresponding xh data.
                     EX: coefmatrix_ = [[10, 20, 15],
                                        [5, 10, 12]]
                                 xh_ = [[31, 26],
                                        [25, 28]]
                         means that [10, 20, 15] are the coefficients of regression which is
                         calculated by [31, 26], and [5, 10, 12] are the coefficients which
                         is calcualted by [25, 28].
        fitted_value_: the fitted value of corresponding xh data.
        """
        X = np.array(X)
        y = np.array(y)
        xh = np.array(xh)
        n, p = X.shape
        self._n = n
        self._p = p
        
        # coef_matrix is the coefficient of every regression coefficient by xh
        coef_matrix = np.zeros((xh.shape[0], X.shape[1]+1))
        # fitted result
        result = np.zeros((xh.shape[0], 1))
        
        for i, ixh in enumerate(xh):
            d = self._distance(X, ixh, isscale)
            dq = np.sort(d, axis=0)[int(n*q)-1]
            
            weight = np.zeros_like(d)
            bool_idx = d < dq
            weight[bool_idx] = (1 - (d[bool_idx]/dq)**3) ** 3
            
            wr = Weighted()
            wr.fit(X, y, weight)
            coef_matrix[i,:] = wr.coefficient_.ravel()
            result[i] = wr.predict(ixh[np.newaxis,:])
        
        ones = np.ones((X.shape[0], 1))
        self.X_ = np.hstack((ones, X))
        self.y_ = y
        self.coefmatrix_ = coef_matrix
        self.fitted_value_ = result
        
    def _distance(self, X, xs, isscale=True):
        """
        find the distance of every points in X to xs
        X is 2-d array with shape=n*p, n is number of sample data and p
        is dimension
        xs is 2-d array with shape=1*p
        isscale means to whether div std on X and xs
        
        EX: X = [[1, 2],
                 [3, 4]]
            xs = [[2, 2]]
            result = [[      1], 
                      [sqrt(5)]]
        """
        X = np.array(X)
        xs = np.array(xs)
        if xs.ndim == 1:
            xs = xs[np.newaxis,:]
            
        if isscale:
            scale = X.std(ddof=1, axis=0)[np.newaxis,:]
            X = X / scale
            xs = xs / scale
        
        return np.sqrt(np.sum((X - xs)**2, axis=1))[:,np.newaxis]
    
    
class RegressionTree:
    def __init__(self):
        pass
    
    def fit(self):
        pass