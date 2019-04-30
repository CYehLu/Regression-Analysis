import numpy as np
from scipy.stats import shapiro, kstest
from scipy.optimize import fmin
import matplotlib.pyplot as plt

from .ols import LinearRegression
from .Inner.distribution import critical_value, find_p_value, gaussian
from .Inner.test_method import _Brown_Forsythe, _Breusch_Pagan


class ResidualAnalysis:
    def __init__(self, regobj):
        # check regobj is a regression obj
        allowed_list = ['LinearRegression', 'SimpleRegression', 'ExpRegression']
        if regobj.__class__.__name__ not in allowed_list:
            raise TypeError(f"'regobj' can not be '{regobj.__class__.__name__}'")
        else:
            self.regobj = regobj
            
        # check residual is exist
        try:
            residual = regobj.residual_.ravel()
            self.residual_ = residual
        except AttributeError as ae:
            raise TypeError("regobj has not fitted yet.")
            
        X = self.regobj.X_
        H = X @ np.linalg.inv(X.T @ X) @ X.T
        hii = H.diagonal()
        MSE = self.regobj.MS_['MSE']
        self.leverage_ = hii
        self.studentized_residual_ = residual / np.sqrt(MSE * (1-hii))
        self.semistudentized_residual_ = residual / np.sqrt(MSE)
        self.deleted_residual_ = residual / (1 - hii)
            
    def PP_plot(self):
        """
        plot a normal probability plot of residual
        
        Return:
        ------
        fig: figure of plot
        except_value: except value under normal distrubution hypothesis
        """
        n = self.regobj.residual_.size
        MSE = self.regobj.MS_['MSE']
        
        temp = np.argsort(self.residual_)
        residual_rank = np.zeros_like(temp)
        residual_rank[temp] = np.arange(1, len(temp)+1)
        
        except_value = np.zeros_like(self.residual_)
        for i, rank in enumerate(residual_rank):
            c = critical_value('gaussian', alpha=1-(rank-0.375)/(n+0.5), mean_var=(0,1))
            except_value[i] = np.sqrt(MSE) * c
            
        fig = plt.figure()
        plt.plot(except_value, self.residual_, 'b.')
        minv, maxv = np.min(except_value), np.max(except_value)
        plt.plot([minv, maxv], [minv, maxv], 'k', alpha=0.1)
        plt.title('normal probability plot')
        plt.xlabel('except value')
        plt.ylabel('residual')
        
        return fig, except_value
    
    def residual_plot(self, xvar='y', yvar='residual'):
        """
        plot residual
        
        Parameter:
        ---------
        xvar: str, 'y' or 'yhat'. 
              'y' mean to y data and 'yhat' mean to predicted data of X data.
        yvar: str, 'residual', 'semi-t' or 't'.
              'residual' mean to residual
              'semi-t' mean to semistudentized residual
              't' mean to studentized residual
        
        Return:
        ------
        fig: figure of plot
        """
        if xvar == 'yhat':
            x = self.regobj.y_.ravel() - self.residual_
            xlabel = '$\^{y}$'
        elif xvar == 'y':
            x = self.regobj.y_
            xlabel = 'y'
        else:
            raise TypeError(f"unavailable xvar: {xvar}")
            
        if yvar == 'residual':
            y = self.residual_
            ylabel = 'residual'
        elif yvar == 'semi-t':
            y = self.semistudentized_residual_
            ylabel = 'semistudentized residual'
        elif yvar == 't':
            y = self.studentized_residual_
            ylabel = 'studentized residual'
        else:
            raise TypeError(f"unavailable yvar: {yvar}")
            
        fig = plt.figure()
        ax = fig.gca()
        plt.plot(x, y, 'b.')
        plt.hlines(0, ax.get_xbound()[0], ax.get_xbound()[1], alpha=0.1)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title('residual plot')
        
        return fig
    
    def constant_variance_test(self, kind='bf', method='both', alpha=0.05):
        """
        test if the residual obey constant variance hypothesis
        
        for 'bf':
            test statistics is tBF.
            if p-value is small or abs(tBF statistics) <= critical value, 
            null hypothesis will be rejected (H0: variance is const)
        for 'bp':
            test statistics is X_BP2
            if p-value is small or X_BP2 <= critical value, 
            null hypothesis will be rejected (H0: variance is const)
        
        Parameter:
        ---------
        kind: 'bf' or 'bp'. mean to Brown Forsythe and Breusch Pagan respectively
              defult is 'bf'
        method: 'both', 'p' or 'region'.
                if p, it will return p-value. 
                if 'region', it will return (test statistics, critical value of reject region)
                if 'both', it will return (p-value, test statistics, critical value)
                defult is 'both'
        alpha: float, significance level. defult is 0.05
        """
        if method not in ['both', 'p', 'region']:
            raise TypeError(f"unavailable method: {method}")
        
        if kind == 'bf':
            y = self.regobj.y_.ravel()
            return _Brown_Forsythe(y, self.residual_, method, alpha)
        elif kind == 'bp':
            # determine whether to drop the first column
            if self.regobj._isintercept:
                X = self.regobj.X_[:,1:]
            else:
                X = self.regobj.X_
            SSE = self.regobj.SS_['SSE']
            return _Breusch_Pagan(X, self.residual_, SSE, method, alpha)
        else:
            raise TypeError(f"unavailable kind: {kind}")
        
    def lack_fit_F_test(self, method='both', alpha=0.05):
        """
        Test if the linear function hypothesis is true
        If p-value is small or F-statistics < critical value,
        the null hypothesis: 'regression function is linear' will be rejected,
        it means that regression function might be nonlinear.
        
        Data should contain deplicate observation points when execute this test.
        
        Parameter:
        ---------
        method: 'both', 'p' or 'region'.
                if p, it will return p-value. 
                if 'region', it will return (test statistics, critical value of reject region)
                if 'both', it will return (p-value, F statistics, critical value)
                defult is 'both'
        alpha: float, significance level. defult is 0.05
        """
        X = self.regobj.X_
        y = self.regobj.y_
        
        X_uniq = np.unique(X, axis=0)
        c = X_uniq.shape[0]  # unique level
        n = X.shape[0]
        if c == n:
            # there are no duplicate data
            raise ValueError("it should contain duplicate observation data.")
        
        # calculate SSPE (pure error sum of square)
        df_SSPE = n - c
        SSPE = 0
        for xuniq in X_uniq:
            y_correspond = y[(X == xuniq).all(axis=1)]
            y_level_mean = y_correspond.mean()
            SSPE += np.sum((y_correspond - y_level_mean) ** 2)
            
        SSE = self.regobj.SS_['SSE']
        SSLF = SSE - SSPE  # SSLF: lack of fit sum of square
        df_SSLF = c - X.shape[1]
        
        # get F statistics
        F = (SSLF / df_SSLF) / (SSPE / df_SSPE)
        
        if method == 'p':
            pvalue = find_p_value('F', 'right', F, dfs=(df_SSLF, df_SSPE))
            return pvalue
        if method == 'region':
            critical = critical_value('F', alpha, dfs=(df_SSLF, df_SSPE))
            return (F, critical)
        elif method == 'both':
            pvalue = find_p_value('F', 'right', F, dfs=(df_SSLF, df_SSPE))
            critical = critical_value('F', alpha, dfs=(df_SSLF, df_SSPE))
            return (pvalue, F, critical)
        else:
            raise TypeError(f"unavailable method: {method}")
            
    def normality_test(self, kind):
        """
        Test whether the residual is obey normal distribution.
        
        Parameter:
        ---------
        kind: str. 'kw' or 'sw', mean to Kolmogorov-Smirnov and Shapiro-Wilk respectively.
        
        Return:
        ------
        tuple of (statistics, pvalue)
        """
        if kind == 'ks':
            MSE = self.regobj.MS_['MSE']
            statistic, pvalue = kstest(self.residual_, lambda x: gaussian(x, 0, MSE))
            return pvalue, statistic
        elif kind == 'sw':
            W, pvalue = shapiro(self.residual_)
            return pvalue, W
    
    def autocorrelation_test(self, residual=None):
        """
        Test whether the residual are autocorrelated by Durbin-Watson test.
        H0: no autocorrelation
        Ha: positive autocorrelation
        
        If DW is large enough (larger than d_U), then there is no autocorrelation.
        But it DW < d_L, the autocorrelation is exist.
        
        DW test is to check if there is 'positive' autocorrelation.
        If want to test 'negative' autocorrelation, the test statistic is 4 - DW,
        then compare 4-DW to d_L and d_U, reject H0 if 4-DW < d_L.
        
        Critical value (d_L, d_U) table:
        http://www.real-statistics.com/statistics-tables/durbin-watson-table/
        
        Parameter:
        ---------
        residual: array-like. Default is None, means that it will use self.residual_
                  to calculate dw statistic.
        
        Return:
        ------
        DW statistic: float
        """
        if residual is None:
            e = self.residual_
        else:
            e = residual
        return np.sum((e[1:]-e[:-1]) ** 2) / np.sum(e**2)
    
    def Cochrane_Orcutt(self, itermax=5):
        """
        Using Cochrane Orcutt procedure to estimate the autocorrelation parameter.
        It should check the regression of Y_t-r*Y_(t-1) to X_t-r*X_(t-1) is no
        autocorrelation by Durbin-Waston test.
       
        Parameter:
        ---------
        itermax: int, the maximum number of iteration. Default is 5.
        
        Return:
        ------
        Tuple (result, models).
        Result is a 2-d array which shape = (itermax, 2).
        The rows mean that the information of every iteration.
        The first column is the estimate r value and the second one is the 
        corresponding Durbin-Waston statistic at this iteration.
        Models is the linear regression modesl of each iteration.
        """
        resid = self.residual_
        X = self.regobj.X_[:,self.regobj._isintercept:]
        y = self.regobj.y_
        
        estimate_r = lambda resid: np.sum(resid[1:]*resid[:-1]) / np.sum(resid[:-1]**2)
        r = estimate_r(resid)
        
        models = []
        result = np.zeros((itermax, 2))
        result[0,0] = r
        result[0,1] = self.autocorrelation_test()
        
        for i in range(itermax):
            X = X[1:,:] - r * X[:-1,:]
            y = y[1:] - r * y[:-1]
            lr = LinearRegression()
            lr.fit(X, y)
            models.append(lr)
            resid = lr.residual_
            
            result[i,0] = r
            result[i,1] = self.autocorrelation_test(resid)
            
            resid = lr.residual_
            r = estimate_r(resid)
            
        return result, models
    
    def Hildreth_Lu(self, rs=None):
        """
        Using Hildreth-Lu procedure to estimate the autocorrelation parameter (r).
        This procedure is to find the best r which can make the regression SSE lowest.
        It should check the regression of Y_t-r*Y_(t-1) to X_t-r*X_(t-1) is no
        autocorrelation by Durbin-Waston test.
        
        Parameter:
        ---------
        rs: array-like. 
            If None, it will find the best r which can make SSE lowest.
            If rs is given, it will find the best r in the rs.
            Default is None.
            
        Return:
        ------
        tuple (r, dw), where r is the autocorrelation parameter and dw is the
        corresponding Durbin-Waston statistic.
        """
        X = self.regobj.X_[:,self.regobj._isintercept:]
        y = self.regobj.y_
        
        if rs is None:
            r = fmin(self._get_transform_SSE, 0.5, args=(X, y), disp=False)[0]
        else:
            SSEs = [self._get_transform_SSE(r, X, y) for r in rs]
            r = rs[np.argmin(SSEs)]
            
        X = X[1:,:] - r * X[:-1,:]
        y = y[1:] - r * y[:-1]
        lr = LinearRegression()
        lr.fit(X, y)
        resid = lr.residual_
        dw = self.autocorrelation_test(resid)
        return (r, dw)
            
    def _get_transform_SSE(self, r, X, y):
        """
        This is used in Hildreth in order to find the best r which
        can make the SSE lowest
        """
        X = X[1:,:] - r * X[:-1,:]
        y = y[1:] - r * y[:-1]
        lr = LinearRegression()
        lr.fit(X, y)
        return lr.SS_['SSE']
        

class Diagnosis:
    def __init__(self, regobj):
        # check regobj is a regression obj
        if regobj.__class__.__name__ not in ['LinearRegression' or 'SimpleRegression']:
            raise TypeError("'regobj' should be 'LinearRegression' or 'SimpleRegression',"
                            + f"not '{regobj.__class__.__name__}'")
        else:
            self.regobj = regobj
            
        X = regobj.X_
        y = regobj.y_
        residual = regobj.residual_
        SSE = self.regobj.SS_['SSE']
        MSE = self.regobj.MS_['MSE']
        n, p = X.shape

        H = X @ np.linalg.inv(X.T @ X) @ X.T
        hii = H.diagonal()[:,np.newaxis]
        
        self.residual_ = residual
        self.studentized_residual_ = residual / np.sqrt(MSE * (1-hii))
        self.semistudentized_residual_ = residual / np.sqrt(MSE)
        self.deleted_residual_ = residual / (1 - hii)
        self.studentized_deleted_residual_ = residual * np.sqrt((n-p-1) / (SSE*(1-hii)-residual**2))
        self.X_ = X
        self.y_ = y
        self._H = H
        self._hii = hii
    
    def added_variable_plot(self, origin_var, added_var, return_data=False):
        """
        plot added variable plot (or partial regression plot, adjusted variable plot)
        
        Parameter:
        ---------
        origin_var: list with int element. 
                    indicate which variables would contain in model first.
        added_var: scaler. it mean to which variable would be added.
        return_data: bool, determine whether to return residual data
        
        Return:
        ------
        if return_df = True:
        return (fig, residual_origin, residual_added)
        where residual_origin is the residual of y to original-variable regression,
        and residual_added is the residual of added-variable to original variable regression.
        
        if return_df = False:
        return fig
        """
        X = self.X_[:,self.regobj._isintercept:]   # remove intercept column
        y = self.y_
        
        X_origin = X[:,origin_var]
        X_added = X[:,[added_var]]
        
        lr_origin = LinearRegression(intercept=self.regobj._isintercept)
        lr_origin.fit(X_origin, y)
        residual_origin = lr_origin.residual_
        
        lr_added = LinearRegression(intercept=self.regobj._isintercept)
        lr_added.fit(X_origin, X_added)
        residual_added = lr_added.residual_
        
        fig = plt.figure()
        plt.plot(residual_added, residual_origin, 'b.')
        
        Xori_str = ','.join([f'$X_{v}$' for v in origin_var])
        xlabel = f'e($X_{added_var}$|{Xori_str})'
        ylabel = f'e(Y|{Xori_str})'
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title('Added variable plot')
        
        if return_data:
            return fig, residual_origin, residual_added
        else:
            return fig
        
    def outlier_detect_y(self, alpha=0.05):
        """
        Identifying outlying Y observations by studentized deleted residual.
        
        The decision rule is: if abs(studentized deleted residual) is larger than
        critical value, the observation point will be seen as a y outlier.
        
        But it recomments that if an observation point is be seen not a y outlier,
        it is still necessary to check whether this observation will affect regression.
        Because it is a Bon ferroni process, and Bon ferroni takes a more conservative 
        attitude.
        
        Parameter:
        ---------
        alpha: float, significance level. defult is 0.05
        
        Return:
        ------
        numpy 2-d matrix, shape = (n, 3) where n is sample size.
        column 1 is the studentized deleted residual,
        column 2 is the critical value base on given alpha, if abs(studentized deleted
        residual) is larger than critical value, the point will be seen as an outlier.
        column 3 is p-value, if p-value is small enough, then the point will be seen as 
        an outlier.
        """
        n, p = self.X_.shape
        
        # 3 cols: studentized deleted residual, critical value, p-value
        result = np.zeros((n,3))
        result[:,0] = self.studentized_deleted_residual_.ravel()
        result[:,1] = critical_value('t', alpha/(2*n), dfs=(n-p-1))
        
        findp = lambda statistics: find_p_value('t', 'two', statistics, dfs=(n-p-1))
        result[:,2] = np.apply_along_axis(findp, 0, result[:,0])
        
        return result
    
    def outlier_detect_x(self):
        """
        Identifying outlying X observations by hat matrix leverage value.
        
        There are some methods to suggest how to use leverage value to determine
        which observation point is an x outlier:
        (1) If a leverage value is large than twice of average leverage value,
        (2) If a leverage value is over 0.5,
        (3) If a leverage value is much higher than the other leverage value
        and these methods can be used in the case which variables and observation
        points are not too small.
            
        Return:
        ------
        hii: 2-d array, hat matrix leverage value. it shape = (n, 1) where n is the
             number of observation points.
        """
        return self._hii
    
    def extrapolation_detect(self, xs):
        """
        Identifying whether the x point has extrapolation by hat matrix.
        
        If the return hii value is in the range of data hii values, it has no problem of
        extrapolation. But if return hii value is too high (much larger than data hii value),
        it means that the input data contain the extrapolation problem.
        
        Parameter:
        ---------
        xs: 2-d array-like. The rows mean to observation point and columns mean to variables.
        
        Return:
        ------
        (hii_range, hii_xs).
        hii_range is a 1*2 array which represent the lower and upper bound of data hii range,
        hii_xs is a 2-d array with shape = (n, 1), which represent the hii value of input xs
        and n is the number of observation points.
        """
        hii_range = np.array([self._hii.min(), self._hii.max()])
        
        xs = np.array(xs)
        ones = np.ones((xs.shape[0], 1))
        xs = np.hstack((ones, xs))
        
        hii_xs = np.zeros((xs.shape[0], 1))
        X = self.X_
        for i, x in enumerate(xs):
            hii_xs[i] = x.T @ np.linalg.inv(X.T @ X) @ x
            
        return hii_range, hii_xs
    
    def influential_detect(self, index=None):
        """
        Identifying influential cases by DFFITS or COOK's distance.
        
        It recommend that use 'detect_outlier_x' and 'detect_outlier_y' to find out which
        points are potential influential outlier, then use this method to understand whether
        the outliers are influential or not.
        If the outliers are not influential, it can not to correct because these outliers
        are not affect the regression model.
        But if the outliers are influential, it should be corrected.
        
        The recommend rule to determine whether the point is influential:
        For DFFITS:
        (1) In small dataset, abs(DFFITS) >= 1
        (2) In large dataset, abs(DFFITS) >= 2*sqrt(p/n), where p is number of variables and
            n is number of observation points.
        For COOK's distance:
        (1) The correspond precentage number of F-distribution >= 0.5, which degree of freedom
            is (p,n-p). It is equivalent to the p-value of left tail F-distribution.
            This rule is the 3rd column of return matrix.
        
        Paramter:
        --------
        index: list of int. It is the index of which observation data will be calculate the
               influence. If None, it will calculate ALL the observation data.
            
        Return:
        ------
        2-d array with column1: DFFITS, column2: COOK's distance and column3: percentage number 
        of F-distribution for COOK's distance.
        Rows mean to observation points.
        """
        n, p = self.X_.shape
        
        if index is None:
            index = [i for i in range(n)]
        
        # DFFITS
        ti_check = self.studentized_deleted_residual_[index,:]
        hii_check = self._hii[index,:]
        DFFITS = ti_check * np.sqrt(hii_check / (1-hii_check))
        
        # COOK's Distance
        residual_check = self.residual_[index,:]
        MSE = self.regobj.MS_['MSE']
        D = residual_check**2 * hii_check / (p * MSE * (1-hii_check)**2)
        
        # percentage in F-distribution of COOK's Distance
        f = lambda d: find_p_value('F', 'left', d, dfs=(p,n-p))
        percents = np.apply_along_axis(f, 0, D)
        
        return np.hstack((DFFITS, D, percents))