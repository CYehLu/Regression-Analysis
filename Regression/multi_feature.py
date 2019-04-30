import numpy as np
import pandas as pd

from .ols import LinearRegression
from .Inner.distribution import find_p_value, critical_value


class VIF:
    def __init__(self, regobj):
        if regobj.__class__.__name__ != 'LinearRegression':
            raise TypeError(f"'regobj' should be 'LinearRegression', not '{regobj.__class__.__name__}'")
        else:
            self.regobj = regobj
            self.X_ = regobj.X_[:,regobj._isintercept:]  # remove intercept term
            self.y_ = regobj.y_
            
    def fit(self):
        """
        Calculate the VIF (variance inflation factor) of each coefficient.
        For those coeffients which VIF are larger than 10, it means that the variables are
        have collinearity problem. And if the mean of VIFs is larger than 1, means that 
        collinearity too.
        
        Return:
        ------
        vif: numpy 1-d array. The order is corresponding to the coefficient (exclude intercept
             term).
             EX: return = array([1.08, 0.9, 0.87]), so
                 VIF of b1 = 1.08
                 VIF of b2 = 0.9
                 VIF of b3 = 0.87
        """
        coef = self.regobj.coefficient_[self.regobj._isintercept:]  # remove intercept term
        
        vif = []
        for i, c in enumerate(coef):
            lr = LinearRegression()
            
            X_k = self.X_[:,[i]]
            X_other = np.delete(self.X_, i, axis=1)
            lr.fit(X_other, X_k)
            vif.append(1 / (1-lr.R2_))
            
        return np.array(vif)

            
class ExtraSS:
    def __init__(self, regobj):
        if regobj.__class__.__name__ != 'LinearRegression':
            raise TypeError(f"'regobj' should be 'LinearRegression', not '{regobj.__class__.__name__}'")
        else:
            self.regobj = regobj
            self.X_ = regobj.X_
            self.y_ = regobj.y_
            SSE = regobj.SS_['SSE']
            SSR = regobj.SS_['SSR']
    
    def SSR_extra(self, added_index, origin_index):
        """
        Calculate the extra regression sum of squares.
        
        Parameter:
        ---------
        added_index: list of int. The elements mean to which variables are added to the model.
        origin_index: list of int. The elements mean to which variables are the origianl model
                      variables.
                      EX: want to calculate SSR(X2,X3|X1), then added_index should be [1, 2] and 
                          origin_index = [0]  (Independent variable: X1, X2, X3, ....)
        
        Retrun:
        ------
        extra SSR     
        """
        origin_index = np.array(origin_index)
        added_index = np.concatenate((np.array(added_index), origin_index))
        
        # if isintercept = True, the first column of self.X_ is intercept and should be avoid
        # to slice
        if self.regobj._isintercept:
            origin_index += 1
            added_index += 1
        
        X_added = self.X_[:,added_index]
        X_origin = self.X_[:,origin_index]
        
        lr_added = LinearRegression()
        lr_added.fit(X_added, self.y_)
        SSR_added = lr_added.SS_['SSR']
        
        lr_origin = LinearRegression()
        lr_origin.fit(X_origin, self.y_)
        SSR_original = lr_origin.SS_['SSR']
        
        return SSR_added - SSR_original
    
    def multicoef_test(self, lastn, method='both', alpha=0.05):
        """
        Test the last n coefficients are zeros or not.
        If p-value is small enough or F-statistics is larger than critical value,
        then H0: 'last n coef are all zeros' will be rejected.
        
        Parameter:
        ---------
        lastn: int. 
               It means to the last n coefficients will be test if they are zeros or not.
        method: str of 'both', 'p' or 'rigion'.
                'p' mean to p-value, 'rigion' mean to find critical value of reject region,
                and 'both' mean to return both p-value and critical value.
        alpha: float. Significance level, defult is 0.05.
        
        Return:
        ------
        If method = 'p', return p-value.
        If method = 'rigion', return tuple (F-statistics, critical value).
        If method = 'both', return tuple (p-value, F-statistics, critical value).
        """
        X = self.X_[:,self.regobj._isintercept:]
        y = self.y_
        n, p = self.X_.shape
        tag = self.regobj._isintercept
        if lastn > p:
            raise ValueError(f"The 'lastn' can not be larger than number of coefficients {p}.")
        
        all_index = [i for i in range(p-tag)]
        added_index = all_index[-lastn:]
        origin_index = all_index[:len(all_index)-lastn]
        SSRextra = self.SSR_extra(added_index, origin_index)
        
        MSE = self.regobj.MS_['MSE']
        
        F = (SSRextra / lastn) / MSE
        
        if method == 'region' or method == 'both':
            critical = critical_value('F', alpha, dfs=(lastn, n-p))
        if method == 'p':
            pvalue = find_p_value('F', 'right', F, dfs=(lastn, n-p))
            return pvalue
        elif method == 'region':
            critical = critical_value('F', alpha, dfs=(lastn, n-p))
            return (F, critical)
        elif method == 'both':
            pvalue = find_p_value('F', 'right', F, dfs=(lastn, n-p))
            critical = critical_value('F', alpha, dfs=(lastn, n-p))
            return (pvalue, F, critical)
    
    def SSRdecomposition_anova(self, return_df=True):
        """
        Make a ANOVA table which contain SSR decomposition.
        
        Parameter:
        ---------
        return_df: bool, determine whether return DataFrame or not.
                   If False, it will print the table directly.
                   
        Return:
        ------
        ANOVA table (if return_df = True)
        """
        X = self.X_[:,self.regobj._isintercept:]
        y = self.y_
        n, p = X.shape
        
        list_X = ['X'+str(i) for i in range(1, p+1)]
        comma = ','
        index = ['X1'] + [f"X{i}|{comma.join(list_X[:i-1])}" for i in range(2, p+1)]
        index = ['Regression'] + index + ['Error', 'Total']
        columns = ['SS', 'DF', 'MS']
        
        SSR = self.regobj.SS_['SSR']
        SSE = self.regobj.SS_['SSE']
        SST = self.regobj.SS_['SST']
        MSR = self.regobj.MS_['MSR']
        MSE = self.regobj.MS_['MSE']
        F = self.regobj.F_
        
        # calculate the extra part
        all_SSR = []
        for i in range(p):
            lr = LinearRegression()
            lr.fit(X[:,:i+1], y)
            all_SSR.append(lr.SS_['SSR'])
        
        exSSR = [all_SSR[0]]
        for i in range(1, p):
            exSSR.append(all_SSR[i] - all_SSR[i-1])
        
        exSSR = np.array([exSSR]).T
        ones = np.ones_like(exSSR)
        result = np.hstack((exSSR, ones, exSSR))
        
        # combine with original anova table
        table = np.vstack((np.array([[SSR, p, MSR]]),
                           result, 
                           np.array([[SSE, n-p-1, MSE],
                                     [SST, n-1, np.nan]])))
        
        # set F and p-value, and clean up
        df = pd.DataFrame(table, index=index, columns=columns)
        df.loc['Regression', 'F'] = F
        df.loc['Regression', 'p-value'] = find_p_value('F', 'right', F, dfs=(p, n-p-1))
        df.replace(np.nan, '', inplace=True)
        df['DF'] = df['DF'].astype(int)
        
        if return_df:
            return df
        else:
            print(df)
    
    def partial_R2(self, added_index, origin_index):
        """
        Calculate coefficient of partial determination.
        
        Parameter:
        ---------
        added_index: array-like or list of int. 
                     The elements mean to which variables are added to the model.
        origin_index: array-like or list of int. 
                      The elements mean to which variables are the origianl model variables.
                      EX: want to calculate R2_{YX3|X1,X2}, then added_index should be [2] and 
                          origin_index = [0, 1]  (Independent variable: X1, X2, X3, ....)
                          
        Return:
        ------
        partial R2
        """
        X = self.X_[:,self.regobj._isintercept:]
        y = self.y_
        added_index = np.array(added_index)
        origin_index = np.array(origin_index)
        
        lr = LinearRegression()
        lr.fit(X[:,origin_index], y)
        SSE_origin = lr.SS_['SSE']
        SSR_extra = self.SSR_extra(added_index, origin_index)
        
        return SSR_extra / SSE_origin