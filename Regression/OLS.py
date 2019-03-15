import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from Regression.distribution import critical_value, find_p_value


#__all__ = ['SimpleRegression', 'LinearRegression', 'PolynomialRegression', 'LogicalRegression']


class SimpleRegression:
    def __init__(self):
        pass
    
    def fit(self, x, y):
        """
        fit data to the simple regression model
        
        Parameter:
        ---------
        x: 1-d ndarray
        y: 1-d ndarray
        """
        # check input data
        if type(x).__module__ != 'numpy' or type(y).__module__ != 'numpy':
            raise TypeError("x or y should be numpy ndarray")
        if x.size != y.size:
            raise ValueError("x and y size should be the same")
        
        # calculate model coefficient b0 and b1
        xmean = x.mean()
        ymean = y.mean()
        b1 = ((x-xmean)*(y-ymean)).sum() / ((x-xmean)**2).sum()
        b0 = ymean - b1 * xmean
        
        # calculate and set attribute
        n = x.size
        y_hat = b0 + b1 * x
        SST = ((y - ymean)**2).sum()
        SSR = ((y_hat - ymean)**2).sum()
        SSE = SST - SSR
        self.coefficient_ = np.array([b0, b1])
        self.SS_ = {'SST': SST, 'SSR': SSR, 'SSE': SSE}
        self.MS_ = {'MSR': SSR, 'MSE': SSE/(n-2)}
        self.F_ = (n-2) * (SSR/SSE)     # F = MSR / MSE
        self.R2_ = SSR / SST
        self.r_ = b1 * x.std() / y.std() 
        self.x_ = x
        self.y_ = y
        self.residual_ = y - y_hat
        # standard residual is as same as in the self.residual_plot()
        h = 1/n + (x-xmean)**2 / ((x-xmean)**2).sum()
        self.standard_residual_ = self.residual_ / (np.sqrt(SSE/(n-2)) * np.sqrt(1-h))
        
        # it can not set intercept in the simple regression, but it is need
        # to set the attribute because of residual analysis
        self._isintercept = False
        
    def predict(self, x):
        """
        predict input x with fitted simple regression model
        
        Parameter:
        ---------
        x: 1-d ndarray, list or scaler
        
        Return:
        ------
        predict value
        """
        b0, b1 = self._check_and_get_parameter()
        xs = self._convert_to_ndarray(x)
        return b0 + b1 * xs
    
    def back_predict(self, y):
        """
        find x values corresponding to the model predict value y
        
        Parameter:
        ---------
        y: 1-d ndarray, list or scaler
        
        Return:
        ------
        back predict value
        """
        b0, b1 = self._check_and_get_parameter()
        ys = self._convert_to_ndarray(y)
        return (ys - b0) / b1
    
    def confidence_interval(self, x, alpha=0.05):
        """
        calculate confidence interval of x
        
        Parameter:
        ---------
        x: ndarray, list or scaler.
        alpha: float. significance level. defult is 0.05
        
        Return:
        ------
        CI values. list [upper_bound, lower_bound], upper/lower_bound is ndarray
        """
        xs = self._convert_to_ndarray(x)
        n = self.x_.size
        MSE = self.MS_['MSE']
        xmean = self.x_.mean()
        
        t_critical = critical_value('t', alpha=alpha/2, dfs=(n-2))
        standard_error = np.sqrt(MSE) * np.sqrt(1/n + (xs-xmean)**2 / ((self.x_-xmean)**2).sum())
        y_hat = self.predict(xs)
        
        upper_bound = y_hat + t_critical * standard_error
        lower_bound = y_hat - t_critical * standard_error
        return [upper_bound, lower_bound]
    
    def predict_interval(self, x, alpha=0.05):
        """
        calculate predict interval of x
        
        Parameter:
        ---------
        x: ndarray, list or scaler.
        alpha: float. significance level. defult is 0.05
        
        Return:
        ------
        predict interval values. 
        list [upper_bound, lower_bound], upper/lower_bound is ndarray
        """
        xs = self._convert_to_ndarray(x)
        n = self.x_.size
        MSE = self.MS_['MSE']
        xmean = self.x_.mean()
        
        t_critical = critical_value('t', alpha=alpha/2, dfs=(n-2))
        standard_error = np.sqrt(MSE) * np.sqrt(1 + 1/n + (xs-xmean)**2 / ((self.x_-xmean)**2).sum())
        y_hat = self.predict(xs)
        
        upper_bound = y_hat + t_critical * standard_error
        lower_bound = y_hat - t_critical * standard_error
        return [upper_bound, lower_bound]
    
    def coefficient_CI(self, alpha=0.05):
        """
        calculate the confident interval (CI) of the coefficients
        
        Parameter:
        ---------
        alpha: float. significance level. defult is 0.05
        
        Return:
        ------
        a list of two tuple. [(CI of b0), (CI of b1)]
        """
        b0, b1 = self.coefficient_
        t_critical = critical_value('t', alpha=alpha/2, dfs=(n-2))[0]  # [0] convert to scaler
        
        try:
            Sb0, Sb1 = self.standard_error_
        except AttributeError:
            # there's no self.standard_error_: do not run coefficient_t_test yet
            self.coefficient_t_test(alpha=(alpha, alpha))
            Sb0, Sb1 = self.standard_error_
        
        return [(b0 - t_critical*Sb0, b0 + t_critical*Sb0), 
                (b1 - t_critical*Sb1, b1 + t_critical*Sb1)]
    
    def plot(self, scatter=True, reg=True, CI=True, PI=False, alpha=0.05):
        """
        plot data
        
        Parameter:
        ---------
        scatter: bool, determine whether to plot points
        reg: bool, determine whether to plot regression line
        CI: bool, determine whether to plot confident interval
        PI: bool, determine whether to plot predict interval
        alpha: float. significance level. defult is 0.05
        
        Return:
        ------
        figure
        """
        fig = plt.figure()
        
        x = self.x_
        y = self.y_
        dx = np.abs(np.diff(x)).mean() / 5
        xs = np.arange(x.min(), x.max()+dx, dx)
        ys = self.predict(xs)
        
        if scatter:
            plt.plot(x, y, 'blue', marker='o', alpha=0.5, ls='None')
        if reg:
            plt.plot(xs, ys, 'black', label='regression')
        if CI:
            #uc, lc = self.confidence_interval(xs, alpha)
            uc, lc = self._confidence_band(xs, alpha)
            plt.plot(xs, uc, 'blue', alpha=0.5, label='confident interval')
            plt.plot(xs, lc, 'blue', alpha=0.5)
        if PI:
            up, lp = self.predict_interval(xs, alpha)
            plt.plot(xs, up, 'red', alpha=0.5, label='predict interval')
            plt.plot(xs, lp, 'red', alpha=0.5)
            
        plt.legend()
        return fig
    
    def F_test(self, method='both', alpha=0.05):
        """
        run the F test of this simple regression model.
        
        if p-value is small enough, or critical value is small than F statistic,
        then H0: b0=b1=0 would be rejected, means that this regression model is
        significantly effective.
        
        Parameter:
        ---------
        method: str. 'p', 'region' or 'both'. 
                'p' means to p-value, 'region' means to rejection region, 
                'both' means that use p-value and rejection region together
        alpha: float, significance level. defult is 0.05
        
        Return:
        ------
        for method = 'p':
        return p-value
        
        for method = 'region':
        return critical value that the H0 would be rejected if F is larger than critical value
        
        for method = 'both':
        return tuple (p-value, critical value)
        """
        if method not in ['p', 'region', 'both']:
            raise ValueError("unavailable method: {}".format(method))
        
        n = self.x_.size
        
        if method == 'p':
            # calculate p-value
            p = find_p_value('F', 'right', self.F_, dfs=(1,n-2))
            return p

        if method == 'region':
            # calculate rejection region
            critical = critical_value('F', alpha, dfs=(1,n-2))
            return critical[0]    # critical is a ndarray with len=1, use [0] to get scaler
        
        if method == 'both':
            p = find_p_value('F', 'right', self.F_, dfs=(1,n-2))
            critical = critical_value('F', alpha, dfs=(1,n-2))
            return (p, critical[0])
    
    def coefficient_t_test(self, H0_num=(0, 0), tail_type=('two', 'two'), method='both', alpha=(0.05, 0.05)):
        """
        run coefficient b0 and b1 t-test
        
        Parameter:
        ---------
        H0_num: tuple, value used in null hypothesis.
                if (x,y), means that H0: b0=x and H0: b1=y.
                default is (0,0)
        tail_type: tuple, tail type.
                 if H0_num = (x,y), and tail_type = ('two', 'right'),
                 means that H1: b0 != x, and H1: b1 > y.
                 there are three options: 'two', 'right' and 'left'
                 default is ('two', 'two')
        method: str. 'p', 'region' or 'both'. 
                'p' means to p-value, 'region' means to rejection region, 
                'both' means that use p-value and rejection region together
        alpha: tuple, float, significance level for b0 and b1. defult is (0.05,0.05)
        
        Return:
        ------
        for method = 'p':
        return (p0, p1), p0 is the p-value for b0 and p1 is for b1
        
        for method = 'region':
        return (critical0, critical1),
        critical value that the H0 would be rejected if t is larger than critical value
        critical0 is the critical value for b0 and critical1 is for b1
        
        for method = 'both':
        return [(p0, p1), (critical0, critical1)]
        """
        # get some information
        b0, b1 = self.coefficient_
        MSE = self.MS_['MSE']
        x = self.x_
        xmean = x.mean()
        n = x.size
        
        # calculate standard error of b0 and b1
        Sb0 = np.sqrt(MSE) * np.sqrt(1/n + xmean**2/((x-xmean)**2).sum())
        Sb1 = np.sqrt(MSE) / np.sqrt(((x-xmean)**2).sum())
        self.standard_error_ = (Sb0, Sb1)
        
        # calculate t statistic
        t0 = (b0 - H0_num[0]) / Sb0
        t1 = (b1 - H0_num[1]) / Sb1
        if H0_num == (0, 0):
            # prevent self.t_ changing with difference coef t-test
            self.t_ = (t0, t1)
        
        # do hypothesis test
        if method == 'p' or method == 'both':
            # calculate p-value
            # b0
            p0 = find_p_value('t', tail_type[0], t0, dfs=(n-2))
            # b1
            p1 = find_p_value('t', tail_type[1], t1, dfs=(n-2))
            
        if method == 'region' or method == 'both':
            # calculate rejection region
            # because critical_value is calculate based on right area,
            # so use c to control
            
            # b0
            if tail_type[0] == 'right':
                alpha0 = alpha[0]
                c = 1
            elif tail_type[0] == 'left':
                alpha0 = alpha[0]
                c = -1    # t-distribution is almost symmetric
            elif tail_type[0] == 'two':
                alpha0 = alpha[0] / 2
                c = 1
            critical0 = critical_value('t', alpha0, dfs=(n-2)) * c

            # b1
            if tail_type[1] == 'right':
                alpha1 = alpha[1]
                c = 1
            elif tail_type[1] == 'left':
                alpha1 = alpha[1]
                c = -1    # t-distribution is almost symmetric
            elif tail_type[1] == 'two':
                alpha1 = alpha[1] / 2
                c = 1
            critical1 = critical_value('t', alpha1, dfs=(n-2)) * c
            
        # return value
        if method == 'p':
            return (p0, p1)
        elif method == 'region':
            return (critical0[0], critical1[0])  # [0] is because critical_value return ndarray
        elif method == 'both':
            return [(p0, p1), (critical0[0], critical1[0])]
        
    def table(self, kind=('coef', 'anova'), return_df=True):
        """
        get regression analysis table
        
        Parameter:
        ---------
        kind: tuple of strings. 'coef' means coefficient table, 'anova' means ANOVA table
        return_df: bool. if True, it will return pandas DataFrame, and if False it will print out
        
        Return:
        ------
        pandas DataFrame (if return_df=True)
        """
        if 'coef' in kind:
            p = self.coefficient_t_test(method='p')
            dfc = pd.DataFrame(np.array([self.coefficient_, self.standard_error_, self.t_, p]).T, 
                               index=['Intercept', 'X'], 
                               columns=['Coefficient', 'Standard Error', 't', 'p-value'])
        if 'anova' in kind:
            n = self.x_.size
            p = self.F_test(method='p')

            data = np.array([[self.SS_['SSR'], 1, self.MS_['MSR'], self.F_, p],
                             [self.SS_['SSE'], n-2, self.MS_['MSE'], np.nan, np.nan],
                             [self.SS_['SST'], n-1, np.nan, np.nan, np.nan]])
            dfa = pd.DataFrame(data, 
                               columns=['SS', 'DF', 'MS', 'F', 'p-value'], 
                               index=['Regression', 'Error', 'Total'])
            dfa.replace(np.nan, '', inplace=True)

        if return_df:
            if 'coef' in kind and 'anova' not in kind:
                return dfc
            elif 'anova' in kind and 'coef' not in kind:
                return dfa
            elif 'anova' in kind and 'coef' in kind:
                return (dfc, dfa)
        else:
            if 'coef' in kind:
                print(dfc)
                print()
            if 'anova' in kind:
                print(dfa)
        
    def residual_plot(self, xaxis='y_hat', yaxis='normal'):
        """
        plot residual distribution
        
        Paramter:
        --------
        xaxis: str. 'y_hat' or 'x', y_hat is the predict value of input x data.
               default is 'y_hat'
        yaxis: str. 'normal' or 'standard'
               normal means normal residual, and standard means standard residual.
               
        Return:
        ------
        figure
        
        
        #######
        method of calculating standard residual is based on '統計學-方法與應用(下)' ISBN: 986-7433-06-8
        """
        if xaxis not in ['y_hat', 'x']:
            raise ValueError("unavailable x axis category")
        if yaxis not in ['normal', 'standard']:
            raise ValueError("unavailable y axis category")
        
        residual = self.residual_
        x = self.x_
        n = x.size
        
        if xaxis == 'y_hat':
            xdata = self.y_ - residual   # y_hat
            xlabel = '$\^{y}$'
        elif xaxis == 'x':
            xdata = x
            xlabel = 'x'
        
        if yaxis == 'normal':
            ydata = residual
            ylabel = 'residual'
        elif yaxis == 'standard':
            ydata = self.standard_residual_
            ylabel = 'standard residual'
            
        fig = plt.figure()
        plt.plot(xdata, ydata, 'b.')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        return fig
    
    def _confidence_band(self, xs, alpha):
        """
        calculate Working-Hotelling confidence band.
        it is used in 'plot' method in order to plot CI.
        """
        xmean = self.x_.mean()
        n = self.x_.size
        MSE = self.MS_['MSE']
        
        ys = self.predict(xs)
        W = np.sqrt(2 * critical_value('F', alpha, dfs=(2,n-2)))
        stderr = np.sqrt(MSE) * np.sqrt(1/n + (xs-xmean)**2 / ((self.x_-xmean)**2).sum())
        return (ys-W*stderr, ys+W*stderr)
        
    def _check_and_get_parameter(self):
        """
        check model parameter b0 and b1 exist or not.
        if b0 and b1 do not exist, raise AttributeError
        else, return b0 and b1
        """
        try:
            return self.coefficient_
        except AttributeError as error:
            raise AttributeError("Unfitted: can not find attribute 'self.coefficient_'")
            
    def _convert_to_ndarray(self, x):
        """
        check data is a numpy ndarray
        if not, convert x to ndarray
        only ndarray, list or scaler available
        """        
        if type(x).__module__ == 'numpy':
            return x
        elif isinstance(x, list):
            return np.array(x)
        elif isinstance(x, (int, float)):
            return np.array([x])
        else:
            raise TypeError("x should be ndarray, list or scaler")
    

class LinearRegression:
    def __init__(self, intercept=True):
        self._isintercept = intercept
    
    def fit(self, X, y):
        """
        fit data to the linear regression model
        
        Parameter:
        ---------
        X: 2-d ndarray. columns of X are variable(features) and rows of X are observation sample
           EX:
             var1 var2 var3
           [[ 1,   2,   3],   sample1
            [ 2,   1,   4],   sample2
            [ 5,   3,   2],   sample3
            [ 7,   1,   2]]   sample4
            
        y: 2-d ndarray. it should be a column vector, and the number of rows is equal to X.shape[0]
           EX:
           [[1],    sample1
            [3],    sample2
            [5],    sample3
            [2]]    sample4
        """
        # check input data
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("X or y should be numpy ndarray")
        if X.shape[0] != y.size:
            raise ValueError("X.shape[0] not equal to y.shape[0]")
        
        X = X.copy()
        n, p = X.shape   # p: X's variables number (without intercept), n: X's sample number
        self._n, self._p = n, p
        
        if self._isintercept:
            X = self._add_intercept(X)
        self.X_ = X
        self.y_ = y
        
        # calculate coefficient
        b = np.linalg.inv(X.T @ X) @ X.T @ y    # @: matrix multiply
        self.coefficient_ = b
        
        # calculate some statistics and set as attributes
        ymean = y.mean() * np.ones(y.shape)
        yhat = X @ b
        SST = ( (y - ymean).T @ (y - ymean) )[0,0]    # [0,0]: convert from array to scaler
        SSR = ( (yhat - ymean).T @ (yhat - ymean) )[0,0]
        SSE = ( (y - yhat).T @ (y - yhat) )[0,0]
        self.SS_ = {'SST': SST, 'SSR': SSR, 'SSE': SSE} 
        self.MS_ = {'MSR': SSR/p, 'MSE': SSE/(n-p-1)}
        self.F_ = self.MS_['MSR'] / self.MS_['MSE']
        self.R2_ = SSR / SST
        self.R2_adj_ = 1 - ((n-1) / (n-p-1)) * (SSE / SST)
        self.residual_ = y - yhat
        
        sb = self.MS_['MSE'] * np.linalg.inv(X.T @ X)
        self.coefficient_standard_error_ = np.sqrt(sb.diagonal())[:, np.newaxis]
        self.t_ = b / self.coefficient_standard_error_
        
    def predict(self, x):
        """
        predict input x with fitted linear regression model
        
        Parameter:
        ---------
        x: 2-d array-like. columns mean to variables(features) and rows mean to 
           observation data which wanted to use to predict
           EX:
             var1 var2 var3
           [[ 1,   2,   3],   sample1
            [ 2,   1,   4],   sample2
            [ 5,   3,   2],   sample3
            [ 7,   1,   2]]   sample4
        
        Return:
        ------
        predict value, it will be a column vector and shape[0] = X.shape[0]
        """
        # if x is not ndarray, convert it to ndarray
        if not isinstance(x, np.ndarray):
            x = np.array(x)
            
        # check variables number (columns of x) is equal to coefficients number
        nvar = x.shape[1] + int(self._isintercept)
        if nvar != self.coefficient_.size:
            raise ValueError("number of variables not matched."
                             + f"it should be {self.coefficient_.size - int(self._isintercept)} variables, "
                             + f"but input is {x.shape[1]}")
            
        # add intercept
        if self._isintercept:
            x = self._add_intercept(x)
        
        return x @ self.coefficient_
    
    def confidence_interval(self, x, alpha=0.05):
        """
        calculate confidence interval of x
        
        Parameter:
        ---------
        x: 2-d array-like. columns mean to variables(features) and rows mean to observation data
           EX:
             var1 var2 var3
           [[ 1,   2,   3],   sample1
            [ 2,   1,   4],   sample2
            [ 5,   3,   2],   sample3
            [ 7,   1,   2]]   sample4
        alpha: float. significance level. defult is 0.05
        
        Return:
        ------
        2-d array with rows mean to difference coefficients and cols means CI
        """
        x = np.array(x)
        y = self.predict(x)
        if self._isintercept:
            x = self._add_intercept(x)
        
        df = self._n - self._p - 1
        t_critical = critical_value('t', alpha/2, dfs=(df))
        
        standard_error = self._calculate_standard_error(x, kind='except')
        
        return np.hstack((y-t_critical*standard_error, y+t_critical*standard_error))
    
    def predict_interval(self, x, alpha=0.05):
        """
        calculate predict interval of x
        
        Parameter:
        ---------
        x: 2-d array-like. columns mean to variables(features) and rows mean to observation data
           EX:
             var1 var2 var3
           [[ 1,   2,   3],   sample1
            [ 2,   1,   4],   sample2
            [ 5,   3,   2],   sample3
            [ 7,   1,   2]]   sample4
        alpha: float. significance level. defult is 0.05
        
        Return:
        ------
        2-d array with rows mean to difference coefficients and cols mean to CI
        """
        x = np.array(x)
        y = self.predict(x)
        if self._isintercept:
            x = self._add_intercept(x)
        
        df = self._n - self._p - 1
        t_critical = critical_value('t', alpha/2, dfs=(df))
        
        standard_error = self._calculate_standard_error(x, kind='predict')
        
        return np.hstack((y-t_critical*standard_error, y+t_critical*standard_error))
    
    def coefficient_CI(self, alpha=0.05):
        """
        calculate the individual confidient interval (CI) of the coefficients
        
        Parameter:
        ---------
        alpha: float. significance level. default is 0.05
        
        Return:
        ------
        ndarray.
        rows mean to difference coefficient, and columns mean to coefficients CI
        """
        n, p = self._n, self._p
        
        t_critical = critical_value('t', alpha/2, dfs=(n-p-1))
        se = self.coefficient_standard_error_
        
        return np.hstack((self.coefficient_ - t_critical*se, self.coefficient_ + t_critical*se))
    
    def coefficient_Bon_ferroni_CI(self, index=None, alpha=0.05):
        """
        calculate the Bon ferroni joint confident interval (CI) of the coefficients
        
        Parameter:
        ---------
        index: list or int.
               if list, the elements of list is the index of coefficient which wanted to 
               form the Bon ferroni CI.
               if int, it will take the first n coefficients to form the Bon ferroni CI.
               if None (default), it will use all coefficients.
        alpha: float. significance level. default is 0.05.
        
        Return:
        ------
        ndarray. 
        rows mean to difference coefficient, and columns mean to Bon ferroni CI
        """
        if isinstance(index, int):
            use_coef = self.coefficient_[:index]
            standard_error_b = self.coefficient_standard_error_[:index]
            g = index
        elif isinstance(index, list):
            use_coef = self.coefficient_[index]
            standard_error_b = self.coefficient_standard_error_[index]
            g = len(index)
        elif index is None:
            use_coef = self.coefficient_
            standard_error_b = self.coefficient_standard_error_
            g = self._p + 1
        else:
            raise TypeError(f"index should be list or int, not {type(index)}.")
        
        B = critical_value('t', alpha/(2*g), dfs=(self._n-self._p-1))
        
        return np.hstack((use_coef - B*standard_error_b, use_coef + B*standard_error_b))
    
    def F_test(self, method='both', alpha=0.05):
        """
        run the F test of this simple regression model.
        
        if p-value is small enough, or critical value is small than F statistic,
        then H0: b0=b1=0 would be rejected, means that this regression model is
        significantly effective.
        
        Parameter:
        ---------
        method: str. 'p', 'region' or 'both'. 
                'p' means to p-value, 'region' means to rejection region, 
                'both' means that use p-value and rejection region together
        alpha: float, significance level. defult is 0.05
        
        Return:
        ------
        for method = 'p':
        return p-value
        
        for method = 'region':
        return critical value that the H0 would be rejected if F is larger than critical value
        
        for method = 'both':
        return tuple (p-value, critical value)
        """
        if method not in ['both', 'p', 'region']:
            raise ValueError("unavailable method: {}".format(method))
            
        n = self._n
        p = self._p
        dfs = (p, n-p-1)
        
        if method == 'p':
            # calculate p-value
            p = find_p_value('F', 'right', self.F_, dfs=dfs)
            return p

        if method == 'region':
            # calculate rejection region
            critical = critical_value('F', alpha, dfs=dfs)
            return critical[0]    # critical is a ndarray with len=1, use [0] to get scaler
        
        if method == 'both':
            p = find_p_value('F', 'right', self.F_, dfs=dfs)
            critical = critical_value('F', alpha, dfs=dfs)
            return (p, critical[0])
    
    def coefficient_t_test(self, method='both', H0_nums=None, alpha=None):
        """
        run coefficients t-test
        
        Parameter:
        ---------
        H0_num: array-like, value used in null hypothesis.
                if None, it will set to be all zeros.
                EX: if (x,y,z), means that H0: b0=x and H0: b1=y and H0: b2=z.
        method: str. 'p', 'region' or 'both'. 
                'p' means to p-value, 'region' means to rejection region, 
                'both' means that use p-value and rejection region together
        alpha: array-like, significance level for coefficients. defult is all 0.05
               it can be ignored if method = 'p'
        
        Return:
        ------
        for method = 'p':
        return a array which elements are the corresponding p-value of coeffieients
        
        for method = 'region':
        return a array which elements are the corresponding critical value of coefficients
        if the t statistics is larger than critical value, the H0 will be rejected
        
        for method = 'both':
        return a 2-d array which first column is the p-value,
        and the second column is the critical value
        """
        n = self._n
        p = self._p
        
        if H0_nums is None:
            H0_nums = np.zeros((p+1,1))
        if alpha is None:
            alpha = 0.05 * np.ones(p+1)
            
        # calculate t statistics
        H0_nums = np.array(H0_nums)
        t = (self.coefficient_ - H0_nums) / self.coefficient_standard_error_
        
        if method == 'p' or method == 'both':
            pvalue = np.zeros((p+1,1))
            for i, ti in enumerate(t):
                pvalue[i,0] = find_p_value('t', 'two', ti, dfs=(n-p-1))
        
        if method == 'region' or method == 'both':
            critical = np.zeros((p+1,1))
            for i, ialpha in enumerate(alpha):
                critical[i,0] = critical_value('t', ialpha, dfs=(n-p-1))
                
        if method == 'p':
            return pvalue
        if method == 'region':
            return critical
        if method == 'both':
            return np.hstack((pvalue, critical))
    
    def table(self, kind=('coef', 'anova'), return_df=True):
        """
        get regression analysis table
        
        Parameter:
        ---------
        kind: tuple of strings. 'coef' means coefficient table, 'anova' means ANOVA table
        return_df: bool. if True, it will return pandas DataFrame, and if False it will print out
        
        Return:
        ------
        pandas DataFrame (if return_df=True)
        """
        if 'coef' in kind:
            coef = self.coefficient_
            se = self.coefficient_standard_error_
            t = self.t_
            pvalue = self.coefficient_t_test(method='p')
            
            index = ['Intercept'] + ['X'+str(i) for i in range(1, self._p+1)]
            columns = ['Coefficient', 'Standard Error', 't', 'p-value']
            dfc = pd.DataFrame(np.hstack((coef, se, t, pvalue)), index=index, columns=columns)

        if 'anova' in kind:
            n = self._n
            p = self._p
            pvalue = self.F_test(method='p')

            data = np.array([[self.SS_['SSR'], p, self.MS_['MSR'], self.F_, pvalue],
                             [self.SS_['SSE'], n-p-1, self.MS_['MSE'], np.nan, np.nan],
                             [self.SS_['SST'], n-1, np.nan, np.nan, np.nan]])
            dfa = pd.DataFrame(data, 
                               columns=['SS', 'DF', 'MS', 'F', 'p-value'], 
                               index=['Regression', 'Error', 'Total'])
            dfa.replace(np.nan, '', inplace=True)
            dfa['DF'] = dfa['DF'].astype(int)

        if return_df:
            if 'coef' in kind and 'anova' not in kind:
                return dfc
            elif 'anova' in kind and 'coef' not in kind:
                return dfa
            elif 'anova' in kind and 'coef' in kind:
                return (dfc, dfa)
        else:
            if 'coef' in kind:
                print(dfc)
                print()
            if 'anova' in kind:
                print(dfa)
    
    def simultaneous_estimation_mean(self, X, method=None, alpha=0.05):
        """
        calculate the simultaneous estimation of mean responses base on input data
        
        Parameter:
        ---------
        X: 2-d array-like. rows mean to difference observation point, 
           columns mean to variables(feature)
        method: str of 'wh' or 'bf'
                'wh' is Working-Hotelling, and 'bf' is Bon ferroni method
                if method is not given, it will choose the tight one
        alpha: float, significance level. defult is 0.05
        
        Return:
        ------
        2-d array which rows mean to observation point and columns mean to lower/upper bound
        """
        n = self._n
        p = self._p
        y = self.predict(X)
        X = self._add_intercept(np.array(X))

        # Working-Hotelling
        W = np.sqrt((p+1) * critical_value('F', alpha, dfs=(p-1, n-p-1)))

        # Bon ferroni
        g = y.size
        B = critical_value('t', alpha/(2*g), dfs=(n-p-1))

        standard_error = self._calculate_standard_error(X, kind='except')

        WH_interval = np.hstack((y-W*standard_error, y+W*standard_error))
        BF_interval = np.hstack((y-B*standard_error, y+B*standard_error))

        if method is None:
            if W <= B:
                return WH_interval
            else:
                return BF_interval
        elif method == 'wh':
            return WH_interval
        elif method == 'bf':
            return BF_interval
        else:
            raise ValueError("unavailable method: {}".format(method))
        
    def simultaneous_estimation_new(self, X, method=None, alpha=0.05):
        """
        calculate the simultaneous prediction intervals for new observation base
        on input data
        
        Parameter:
        ---------
        X: 2-d array-like. rows mean to difference observation point, 
           columns mean to variables(feature)
        method: str of 'wh' or 'bf'
                'wh' is Working-Hotelling, and 'bf' is Bon ferroni method
                if method is not given, it will choose the tight one
        alpha: float, significance level. defult is 0.05
        
        Return:
        ------
        2-d array which rows mean to observation point and columns mean to lower/upper bound
        """
        n = self._n
        p = self._p
        y = self.predict(X)
        X = self._add_intercept(np.array(X))
        g = y.size

        # Scheffe
        S = np.sqrt(g * critical_value('F', alpha, dfs=(g, n-p-1)))

        # Bon ferroni
        B = critical_value('t', alpha/(2*g), dfs=(n-p-1))

        standard_error = self._calculate_standard_error(X, kind='predict')

        SF_interval = np.hstack((y-S*standard_error, y+S*standard_error))
        BF_interval = np.hstack((y-B*standard_error, y+B*standard_error))

        if method is None:
            if S <= B:
                return SF_interval
            else:
                return BF_interval
        elif method == 'sf':
            return SF_interval
        elif method == 'bf':
            return BF_interval
        else:
            raise ValueError("unavailable method: {}".format(method))
            
    def _add_intercept(self, X):
        """
        add intercept at matrix x.
        EX:
        >>> x = np.array([[12, 13, 14],
        ...              [15, 16, 17],
        ...              [18, 19, 20]])
        >>> x = _add_intercept(x)
        >>> x
        np.array([[1, 12, 13, 14],
                  [1, 15, 16, 17],
                  [1, 18, 19, 20]])
        """
        nrows = X.shape[0]
        one = np.ones((nrows, 1))
        return np.hstack((one, X))
    
    def _calculate_standard_error(self, X, kind):
        """
        calculate standard error of each row of X
        
        
        Parameter:
        ---------
        X: 2-d array which rows mean to observation and columns mean to variables
        kind: str of 'except' or 'predict'. 
              means except interval and predict interval respectively
              
        Return:
        ------
        2-d array of standard error which shape = (X rows number, 1)
        """
        if kind == 'except':
            k = 0
        elif kind == 'predict':
            k = 1
        
        MSE = self.MS_['MSE']
        standard_error = np.zeros((X.shape[0], 1))
        for i, xrow in enumerate(X):
            ise = np.sqrt(MSE * (k + xrow @ np.linalg.inv(self.X_.T @ self.X_) @ xrow.T))
            standard_error[i,0] = ise
            
        return standard_error
    
    
class Weighted(LinearRegression):
    def __init__(self, intercept=True):
        self._isintercept = intercept
        
    def fit(self, X, y, weight=None):
        """
        Fit data to weighted linear regression with given weights
        
        Parameter:
        ---------
        X: 2-d array-like. The rows mean to observation data points and the columns
           mean to variables (features).
        y: 2-d array-like. Its shape should be (n*1) where n is the number of samples.
        weight: 1-d or 2-d array-like. The length should be p where p is the number
                of variables (equal to X.shape[1]).
                If None, the weight will be all 1.
        """
        X = np.array(X)
        y = np.array(y)
        
        n, p = X.shape
        self._n = n
        self._p = p
        
        # add intercept
        if self._isintercept:
            ones = np.ones((X.shape[0], 1))
            X = np.hstack((ones, X))
        
        if weight is None:
            W = np.zeros((X.shape[0], X.shape[0]))
            np.fill_diagonal(W, 1)
        else:
            W = np.zeros((X.shape[0], X.shape[0]))
            np.fill_diagonal(W, weight)
        
        # compute coefficient
        b = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ y
        self.coefficient_ = b
        self.weight_ = weight.ravel()
        self.X_ = X
        self.y_ = y
        
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
    
    
class PolynomialRegression(LinearRegression):
    def __init__(self, order, intercept=True):
        self._order = order
        self._isintercept = intercept
        
    def fit(self, x, y):
        """
        Fit data to the polynomial regression model.
        
        Parameter:
        ---------
        x: 1-d array-like. Independent variable.
        y: 1-d array-like. Dependent variable.
        """
        # convert to ndarray and make sure they are 1-d array
        x = np.array(x).ravel()
        y = np.array(y).ravel()
        n = x.size
        
        y = y[:,np.newaxis]
        X = np.zeros((n, self._order))
        for i in range(1, self._order+1):
            X[:,i-1] = x ** i
                
        super().fit(X, y)
        
    
class RidgeRegression():
    def __init__(self):
        self._isintercept = True
    
    def fit(self, X, y, c):
        ones = np.ones((X.shape[0], 1))
        self.X_ = np.hstack((ones, X))
        self.y_ = y
        self.c_ = c
        n, p = X.shape  # p is the number of variables (no intercept term)
        
        # normalize first
        Xn = (X - X.mean(axis=0)) / (X.std(ddof=1, axis=0) * np.sqrt(n-1))
        yn = (y - y.mean()) / (y.std(ddof=1) * np.sqrt(n-1))
        
        # find corrlation matrix and vector
        rXX = np.corrcoef(Xn, rowvar=False)
        rXY = np.zeros((p, 1))
        for i in range(p):
            rXY[i] = np.corrcoef(Xn[:,i], yn.ravel())[0,1]
            
        # calculate coefficients
        coeff = np.linalg.inv(rXX + c*np.eye(p)) @ rXY
        self.beta_coefficient_ = coeff
        
        self.coefficient_ = np.zeros((p+1, 1))
        sy = y.std(ddof=1)
        sx = X.std(ddof=1, axis=0)[:,np.newaxis]
        self.coefficient_[1:] = (sy/sx) * coeff
        self.coefficient_[0] = y.mean() - np.sum(self.coefficient_[1:].ravel() * X.mean(axis=0))
            
        # set other attributes
        ymean = y.mean() * np.ones(y.shape)
        yhat = self.X_ @ self.coefficient_
        SST = ( (y - ymean).T @ (y - ymean) )[0,0]    # [0,0]: convert from array to scaler
        SSR = ( (yhat - ymean).T @ (yhat - ymean) )[0,0]
        SSE = ( (y - yhat).T @ (y - yhat) )[0,0]
        self.SS_ = {'SST': SST, 'SSR': SSR, 'SSE': SSE} 
        self.MS_ = {'MSR': SSR/p, 'MSE': SSE/(n-p-1)}
        self.F_ = self.MS_['MSR'] / self.MS_['MSE']
        self.R2_ = SSR / SST
        self.R2_adj_ = 1 - ((n-1) / (n-p-1)) * (SSE / SST)
        self.residual_ = y - yhat
        
        sb = self.MS_['MSE'] * np.linalg.inv(self.X_.T @ self.X_)
        self.coefficient_standard_error_ = np.sqrt(sb.diagonal())[:, np.newaxis]
        self.t_ = self.coefficient_ / self.coefficient_standard_error_


class LogicalRegression:
    # OLS?????
    pass