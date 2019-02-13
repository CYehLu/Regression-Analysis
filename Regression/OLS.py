import numpy as np
import matplotlib.pyplot as plt

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
        self.coefficient_ = np.array([b0, b1])
        
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
        x = self.x_
        xmean = x.mean()
        n = x.size
        t_critical = critical_value('t', alpha=alpha/2, dfs=(n-2))[0]  # [0] convert to scaler
        
        MSE = self.MS_['MSE']
        Sb0 = np.sqrt(MSE) * np.sqrt(1/n + xmean**2 / ((x-xmean)**2).sum())
        Sb1 = np.sqrt(MSE / ((x-xmean)**2).sum())
        
        b0, b1 = self.coefficient_
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
            uc, lc = self.confidence_interval(xs, alpha)
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
            S = np.sqrt((residual**2).sum() / (n-2))
            h = 1/n + (x-x.mean())**2 / ((x-x.mean())**2).sum()
            ydata = residual / (S*np.sqrt(1-h))   # standard residual
            ylabel = 'standard residual'
            
        fig = plt.figure()
        plt.plot(xdata, ydata, 'b.')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        return fig
        
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
    def __init__(self):
        pass
    
    
class PolynomialRegression:
    def __init__(self):
        pass
    

class LogicalRegression:
    # OLS?????
    pass