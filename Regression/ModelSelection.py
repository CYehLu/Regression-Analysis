from itertools import combinations
from warnings import warn

import numpy as np
import pandas as pd 

from .OLS import LinearRegression
from .distribution import critical_value, find_p_value


def test_reduce_model(X, y, reduce_idx, alpha=0.05, isintercept=True):
    """
    Test if the reduce model can be accept.
    Reduce model is the model which is as same as full model except some 
    coefficients are zero.
    EX: full model -> Y = b0 + b1*X1 + b2*X2 + b3*X4 + error
        reduce model -> Y = b0 + b1*X1 + error
        thus H0: b2=b3=0
             Ha: b2!=0 or b3!=0
             
    Parameter:
    ---------
    X: 2-d ndarray.
    y: 2-d ndarray with shape = (n,1).
    reduce_idx: list, the elements of list is the column index in X which is
                reduced.
                It should be noticed that index is start from 0. So if want to
                test b2=b3=0 (corresponding X column index is 1 and 2), the
                reduce_idx = [1, 2].
    alpha: float, significance level. Default is 0.05.
    isintercept: bool, fit intercept term or not.
    
    Return:
    ------
    array([p, F, critical]).
    'p' is the p-value of reduce model. If p is small enough, H0 will be rejected
    and full model will be accepted.
    'F' is the F statistic of reduce model. It F is larger than critical value,
    H0 will be rejected and full model will be accepted.
    'critical' is the critical value of F statistical.
    """
    n, p = X.shape
    
    # full model
    lr = LinearRegression(isintercept)
    lr.fit(X, y)
    SSE_F = lr.SS_['SSE']
    df_F = n - lr.X_.shape[1]
    
    # reduce model
    X_r = np.delete(X, reduce_idx, axis=1)
    lr.fit(X_r, y)
    SSE_R = lr.SS_['SSE']
    df_R = n - lr.X_.shape[1]
    
    F = ((SSE_R-SSE_F)/(df_R-df_F)) / (SSE_F/df_F)
    critical = critical_value('F', alpha, dfs=(len(reduce_idx),df_F))
    p = find_p_value('F', 'right', F, dfs=(len(reduce_idx),df_F))
    return np.array([p, F, critical])


class Criteria:
    def __init__(self, X, y, isintercept=True):
        self.X_ = X    # this X is the original data matrix, no intercept term
        self.y_ = y
        self._isintercept = isintercept
        
    def find_best(self, kind):
        """
        Find the best criteria value by given kind and its corresponding variables.
        
        *R2 and SSE are not included in this method because they must be lower with higher 
         number of variables. If want to determine which model is better by R2 and SSE, it
         recommend to plot a line chart and find the 'elbow point'.
        
        Parameter:
        ---------
        kind: str, 'R2adj', 'Cp', 'AIC', 'SBC' or 'PRESS'.
              'R2adj' -> adjusted coefficient of multiple determination
              'Cp'    -> Mallows' Cp
              'AIC'   -> Akaike's information criterion
              'SBC'   -> Schwarz' Bayesian criterion
              'PRESS' -> prediction sum of squares
              
        Return:
        ------
        (bestv, bestsub), where 'bestv' is the best (min or max) value of given criteria
        kind and 'bestsub' is the corresponding variables.
        
        Note that the variables in 'bestsub' are start from 1. EX: if there are p variables,
        then all variables are (X1, X2, ..., Xp).
        """
        all_subsets, result = self.all_possible()
        
        kind2index = {'R2adj': 3, 'Cp': 4, 'AIC': 5, 'SBC': 6, 'PRESS': 7}
        
        if kind == 'R2adj':
            kindex = kind2index[kind]
            bestv = np.max(result[:,kindex])
            bestsub = all_subsets[np.argmax(result[:,kindex])]
        elif kind in ['Cp', 'AIC', 'SBC', 'PRESS']:
            kindex = kind2index[kind]
            bestv = np.min(result[:,kindex])
            bestsub = all_subsets[np.argmin(result[:,kindex])]
        else:
            raise TypeError(f"Unavailable kind: {kind}.")
        return bestv, bestsub
        
    def all_possible(self, return_kind='array'):
        """
        Parameter:
        ---------
        return_kind: str, 'array' or 'df'.
                     Determine which kind would be return.
        
        Return:
        ------
        If return_kind = 'array',
        it will return (all subsets, result), where rows of result are difference subsets
        and columns are 'parameters', 'SSE', 'R2', 'R2 adj', 'Cp', 'AIC', 'SBC' and 'PRESS' respectively.
        The i'th element in all subsets is corresponding to the i'th row in result.
        
        If return_kind = 'df',
        it will return a DataFrame which value is as same as the 'result' when return_kind = 'array',
        and the index of DataFrame are the 'all subsets' in which return_kind = 'array'.
        """
        p = self.X_.shape[1]
        all_index = [i for i in range(p)]
        
        # all_subsets will contain every subset of variables with index form.
        # EX: when p=3 (3 variables), 
        # then all_subsets = [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
        all_subsets = []
        for nvar in range(1, p+1):
            all_subsets.extend(list(combinations(all_index, nvar)))
        
        result = np.zeros((len(all_subsets), 8))
        result[:,0] = list(map(lambda index: len(index)+1, all_subsets))
        result[:,1] = list(map(self.SSE, all_subsets))
        result[:,2] = list(map(self.R2, all_subsets))
        result[:,3] = list(map(self.R2_adj, all_subsets))
        result[:,4] = list(map(self.Mallow_Cp, all_subsets))
        result[:,5] = list(map(self.AIC, all_subsets))
        result[:,6] = list(map(self.SBC, all_subsets))
        result[:,7] = list(map(self.PRESS, all_subsets))
        
        # convert from index form to str form, EX: (0, 1) -> ('X1', 'X2')
        all_subsets = [tuple('X'+str(i+1) for i in subset) for subset in all_subsets]
        
        if return_kind == 'array':
            return all_subsets, result
        elif return_kind == 'df':
            df = pd.DataFrame(result)
            df.columns = ['parameters', 'SSE', 'R2', 'R2 adj', 'Cp', 'AIC', 'SBC', 'PRESS']
            df.index = all_subsets
            df['parameters'] = df['parameters'].astype('int')
            return df
        
    def R2(self, index=None):
        """
        Calculate coefficient of determination.
        
        Parameter:
        ---------
        index: list, the variables which want to used to calculate R2.
               If None, it will use all variables to calculate. Default is None.
               EX: If want to use (X1, X3, X4), then index = [0, 2, 3]
               
        Return:
        ------
        Coefficient of determination.
        """
        if index is None:
            index = [i for i in range(self.X_.shape[1])]
        
        X = self.X_[:,index]
        y = self.y_
        
        lr = LinearRegression(intercept=self._isintercept)
        lr.fit(X, y)
        return lr.R2_
    
    def SSE(self, index=None):
        """
        Calculate error sum of squares.
        
        Parameter:
        ---------
        index: list, the variables which want to used to calculate SSE.
               If None, it will use all variables to calculate. Default is None.
               EX: If want to use (X1, X3, X4), then index = [0, 2, 3]
               
        Return:
        ------
        Error sum of squares.
        """
        if index is None:
            index = [i for i in range(self.X_.shape[1])]
        
        X = self.X_[:,index]
        y = self.y_
        
        lr = LinearRegression(intercept=self._isintercept)
        lr.fit(X, y)
        return lr.SS_['SSE']
    
    def R2_adj(self, index=None):
        """
        Calculate adjusted coefficient of multiple determination.
        
        Parameter:
        ---------
        index: list, the variables which want to used to calculate R2_adj.
               If None, it will use all variables to calculate. Default is None.
               EX: If want to use (X1, X3, X4), then index = [0, 2, 3]
               
        Return:
        ------
        Adjusted coefficient of multiple determination.
        """
        if index is None:
            index = [i for i in range(self.X_.shape[1])]
        
        X = self.X_[:,index]
        y = self.y_
        
        lr = LinearRegression(intercept=self._isintercept)
        lr.fit(X, y)
        return lr.R2_adj_
    
    def Mallow_Cp(self, index=None):
        """
        Calculate Mallow's Cp.
        
        Parameter:
        ---------
        index: list, the variables which want to used to calculate Cp.
               If None, it will use all variables to calculate. Default is None.
               EX: If want to use (X1, X3, X4), then index = [0, 2, 3]
               
        Return:
        ------
        Mallow's Cp.
        """
        if index is None:
            index = [i for i in range(self.X_.shape[1])]
            
        n = self.X_.shape[0]
        p = self.X_.shape[1] + 1
        all_index = [i for i in range(self.X_.shape[1])]
        
        MSE_all = self.SSE(all_index) / (n-p)
        SSE_p = self.SSE(index)
        Cp = SSE_p / MSE_all - (n - 2*(len(index)+self._isintercept))
        return Cp
    
    def AIC(self, index=None):
        """
        Calculate Akaike's information criterion.
        
        Parameter:
        ---------
        index: list, the variables which want to used to calculate AIC.
               If None, it will use all variables to calculate. Default is None.
               EX: If want to use (X1, X3, X4), then index = [0, 2, 3]
               
        Return:
        ------
        Akaike's information criterion.
        """
        if index is None:
            index = [i for i in range(self.X_.shape[1])]
            
        n = self.X_.shape[0]
        aic = n * np.log(self.SSE(index)) - n * np.log(n) + 2*(len(index)+1)
        return aic
    
    def SBC(self, index=None):
        """
        Calculate Schwarz' Bayeisan criterion.
        
        Parameter:
        ---------
        index: list, the variables which want to used to calculate SBC.
               If None, it will use all variables to calculate. Default is None.
               EX: If want to use (X1, X3, X4), then index = [0, 2, 3]
               
        Return:
        ------
        Schwarz' Bayeisan criterion.
        """
        if index is None:
            index = [i for i in range(self.X_.shape[1])]
        
        n = self.X_.shape[0]
        sbc = n * np.log(self.SSE(index)) - n * np.log(n) + np.log(n)*(len(index)+1)
        return sbc
    
    def PRESS(self, index=None):
        """
        Calculate prediction sum of squares.
        
        Parameter:
        ---------
        index: list, the variables which want to used to calculate PRESS.
               If None, it will use all variables to calculate. Default is None.
               EX: If want to use (X1, X3, X4), then index = [0, 2, 3]
               
        Return:
        ------
        prediction sum of squares.
        """
        if index is None:
            index = [i for i in range(self.X_.shape[1])]
            
        lr = LinearRegression(intercept=self._isintercept)
        lr.fit(self.X_[:,index], self.y_)
        
        X = lr.X_
        residual = lr.residual_.ravel()
        H = X @ np.linalg.inv(X.T @ X) @ X.T
        hii = H.diagonal()
        press = ((residual / (1 - hii))**2).sum()
        return  press
    

class SearchProcedure:
    def __init__(self, X, y):
        self.X_ = X
        self.y_ = y
        
    def search(self, method, alpha_in=None, alpha_out=None, itermax=None):
        """
        Search procedure that is try to find the best model.
        
        Parameter:
        ---------
        method: str, 'fws', 'bws', 'fw' or 'bw'.
                'fws' -> Forward Stepwise
                'bws' -> Backward Stepwise
                'fw' -> Forward Selection
                'bw' -> Backward Elimination
        alpha_in: float. Significant level of allowing variable added in the model.
        alpha_out: float. Significant level of determine whether variable deleted or not.
                   *It should larger than 'alpha_in'.
        itermax: int. The maximum number of iteration. If None, default is 50.
        
        Return:
        ------
        best_var: tuple. Every elements of 'best_var' is the index of variables which are
                  selected into the model.
        df: pandas DataFrame. It contains the iteration history and the t and p-values.
        """
        # check input parameter
        if method in ['fws', 'bws'] and alpha_in > alpha_out:
            warn("'alpha_out' should larger than 'alphpa_in' to avoid variables"
                 + " that are constantly being added and then shaved.")
        if itermax is None:
            itermax = 50
            
        # start to search
        if method == 'fws':
            best_var, df = self._forward_stepwise(self.X_, self.y_, alpha_in, alpha_out, itermax)
            
        elif method == 'bws':
            best_var, df = self._backward_stepwise(self.X_, self.y_, alpha_in, alpha_out, itermax)
            
        elif method == 'fw':
            best_var, df = self._forward(self.X_, self.y_, alpha_in, itermax)
            
        elif method == 'bw':
            best_var, df = self._backward(self.X_, self.y_, alpha_out, itermax)
            
        return best_var, df
            
    def _forward_stepwise(self, X, y, alpha_in, alpha_out, itermax=50):
        n, p = X.shape
        multi_index = [np.arange(itermax).repeat(2), np.array(['t', 'p']*itermax)]
        df = pd.DataFrame(np.nan * np.zeros((itermax*2 ,p+4)), index=multi_index)
        
        added_var = []
        other_var = list(range(p))
        
        iter_ = 0
        while True:
            if iter_ >= itermax:
                warn("Stop by iteration reach 'itermax'. May not find the best result yet.")
                break
            
            # add variable
            tag, df = self._add(X, y, added_var, other_var, alpha_in, df, iter_)
            if not tag:
                break
            if iter_ == 0:
                iter_ += 1
                continue
            iter_ += 1
                
            # check and remove variable
            tag, df = self._out(X, y, added_var, other_var, alpha_out, df, iter_)
            if tag:
                iter_ += 1
            
        # clean df
        df = df.loc[:iter_-1, :]
        column1 = ['X'+str(c+1) for c in df.columns[:p]]
        column2 = ['R2', 'R2adj', 'Cp', 'result']
        df.columns = column1 + column2
        df = df.replace(np.nan, '-')
        df.loc[:, column2] = df.loc[:, column2].replace('-', ' ')
        df = df.applymap(lambda x: f'{x:.5f}' if isinstance(x, (float)) else x)
        return added_var, df
    
    def _backward_stepwise(self, X, y, alpha_in, alpha_out, itermax=50):
        n, p = X.shape
        multi_index = [np.arange(itermax).repeat(2), np.array(['t', 'p']*itermax)]
        df = pd.DataFrame(np.nan * np.zeros((itermax*2 ,p+4)), index=multi_index)
        
        added_var = list(range(p))
        other_var = []
        
        # calculate the first round
        lr = LinearRegression()
        lr.fit(X, y)
        df.loc[(0,'t'), added_var] = lr.t_[1:].ravel()
        lambda_p = lambda t: find_p_value('t', 'two', t, dfs=(n-p-1))
        df.loc[(0,'p'), added_var] = df.loc[(0,'t'), added_var].map(lambda_p)
        
        cr = Criteria(X, y)
        df.loc[(0,'t'), p] = cr.R2(index=added_var)
        df.loc[(0,'t'), p+1] = cr.R2_adj(index=added_var)
        df.loc[(0,'t'), p+2] = cr.Mallow_Cp(index=added_var)
        df.loc[(0,'t'), p+3] = 'Add all'
        
        # do the other rounds
        iter_ = 1
        while True:
            if iter_ >= itermax:
                warn("Stop by iteration reach 'itermax'. May not find the best result yet.")
                break
            
            tag, df = self._out(X, y, added_var, other_var, alpha_out, df, iter_)
            iter_ += 1
            if not tag:
                break
                
            tag, df = self._add(X, y, added_var, other_var, alpha_in, df, iter_)
            if tag:
                iter_ += 1
                
        # clean df
        df = df.loc[:iter_-2, :]
        column1 = ['X'+str(c+1) for c in df.columns[:p]]
        column2 = ['R2', 'R2adj', 'Cp', 'result']
        df.columns = column1 + column2
        df = df.replace(np.nan, '-')
        df.loc[:, column2] = df.loc[:, column2].replace('-', ' ')
        df = df.applymap(lambda x: f'{x:.5f}' if isinstance(x, (float)) else x)
        return added_var, df
        
    
    def _forward(self, X, y, alpha_in, itermax=50):
        n, p = X.shape
        multi_index = [np.arange(itermax).repeat(2), np.array(['t', 'p']*itermax)]
        df = pd.DataFrame(np.nan * np.zeros((itermax*2 ,p+4)), index=multi_index)
        
        added_var = []
        other_var = list(range(p))
        
        iter_ = 0
        while True:
            if iter_ >= itermax:
                warn("Stop by iteration reach 'itermax'. May not find the best result yet.")
                break
            
            tag, df = self._add(X, y, added_var, other_var, alpha_in, df, iter_)
            iter_ += 1
            
            if not tag:
                break
                
        # clean df
        df = df.loc[:iter_-2, :]
        column1 = ['X'+str(c+1) for c in df.columns[:p]]
        column2 = ['R2', 'R2adj', 'Cp', 'result']
        df.columns = column1 + column2
        df = df.replace(np.nan, '-')
        df.loc[:, column2] = df.loc[:, column2].replace('-', ' ')
        df = df.applymap(lambda x: f'{x:.5f}' if isinstance(x, (float)) else x)
        return added_var, df
    
    def _backward(self, X, y, alpha_out, itermax=50):
        n, p = X.shape
        multi_index = [np.arange(itermax).repeat(2), np.array(['t', 'p']*itermax)]
        df = pd.DataFrame(np.nan * np.zeros((itermax*2 ,p+4)), index=multi_index)
        
        added_var = list(range(p))
        other_var = []
        
        # calculate the first round
        lr = LinearRegression()
        lr.fit(X, y)
        df.loc[(0,'t'), added_var] = lr.t_[1:].ravel()
        lambda_p = lambda t: find_p_value('t', 'two', t, dfs=(n-p-1))
        df.loc[(0,'p'), added_var] = df.loc[(0,'t'), added_var].map(lambda_p)
        
        cr = Criteria(X, y)
        df.loc[(0,'t'), p] = cr.R2(index=added_var)
        df.loc[(0,'t'), p+1] = cr.R2_adj(index=added_var)
        df.loc[(0,'t'), p+2] = cr.Mallow_Cp(index=added_var)
        df.loc[(0,'t'), p+3] = 'Add all'
        
        # do the other rounds
        iter_ = 1
        while True:
            if iter_ >= itermax:
                warn("Stop by iteration reach 'itermax'. May not find the best result yet.")
                break
            
            tag, df = self._out(X, y, added_var, other_var, alpha_out, df, iter_)
            iter_ += 1
            
            if not tag:
                break
                
        # clean df
        df = df.loc[:iter_-2, :]
        column1 = ['X'+str(c+1) for c in df.columns[:p]]
        column2 = ['R2', 'R2adj', 'Cp', 'result']
        df.columns = column1 + column2
        df = df.replace(np.nan, '-')
        df.loc[:, column2] = df.loc[:, column2].replace('-', ' ')
        df = df.applymap(lambda x: f'{x:.5f}' if isinstance(x, (float)) else x)
        return added_var, df
        
    def _out(self, X, y, added_var, other_var, alpha_out, df, iter_):
        """
        Take out a variable from added_var to other_var based on given information.
        It will return the tag and df, which tag is mean to success to delete a variable
        from model if tag is True, otherwise is False.
        And df is the updated df.
        
        The result of added_var and other_var is modified inplace.
        """
        n, p = X.shape
        t_out_critical = critical_value('t', alpha_out, dfs=(n-len(added_var)-1))
        
        lr = LinearRegression()
        lr.fit(X[:,added_var], y)
        
        if np.all(np.abs(lr.t_) > t_out_critical):
            return False, df
        else:
            out_var_arg = np.argmin(lr.t_[1:])
            other_var = other_var.append(added_var[out_var_arg])
            out_var = added_var.pop(out_var_arg)
            
            lr = LinearRegression()
            lr.fit(X[:,added_var], y)
            df.loc[(iter_,'t'), added_var] = lr.t_[1:].ravel()
            lambda_p = lambda t: find_p_value('t', 'two', t, dfs=(n-len(added_var)-1))
            df.loc[(iter_,'p'), added_var] = df.loc[(iter_,'t'), added_var].map(lambda_p)
            
            cr = Criteria(X, y)
            df.loc[(iter_,'t'), p] = cr.R2(index=added_var)
            df.loc[(iter_,'t'), p+1] = cr.R2_adj(index=added_var)
            df.loc[(iter_,'t'), p+2] = cr.Mallow_Cp(index=added_var)
            df.loc[(iter_,'t'), p+3] = 'Remove X' + str(out_var+1) 
            return True, df
    
    def _add(self, X, y, added_var, other_var, alpha_in, df, iter_):
        """
        Take in a variable from other_var to added_var based on given information.
        It will return the tag and df, which tag is mean to success to add a variable
        from model if tag is True, otherwise is False.
        And df is the updated df.
        
        The result of added_var and other_var is modified inplace.
        """
        n, p = X.shape
        
        candidates = [added_var + [ivar] for ivar in other_var]
        
        # find and add variable into model
        pvalues = []
        ts = []
        lrs = []    # models
        for i, cand in enumerate(candidates):
            lr = LinearRegression()
            lr.fit(X[:,cand], y)
            ts.append(lr.t_[-1,0])    # the last coef is from 'other_var'
            pvalues.append(find_p_value('t', 'two', lr.t_[-1], dfs=(n-len(cand)-1))[0])
            lrs.append(lr)
            
        if np.all(np.array(pvalues) > alpha_in):
            # no variable can be added
            return False, df
        else:
            var = candidates[np.argmin(pvalues)][-1]
            added_var.append(var)
            other_var.remove(var)
            
            df.loc[(iter_,'t'), added_var] = lrs[np.argmin(pvalues)].t_.ravel()[1:]
            lambda_p = lambda t: find_p_value('t', 'two', t, dfs=(n-len(cand)-1))
            df.loc[(iter_,'p'), added_var] = df.loc[(iter_,'t'), added_var].map(lambda_p)
            
            cr = Criteria(X, y)
            df.loc[(iter_,'t'), p] = cr.R2(index=added_var)
            df.loc[(iter_,'t'), p+1] = cr.R2_adj(index=added_var)
            df.loc[(iter_,'t'), p+2] = cr.Mallow_Cp(index=added_var)
            df.loc[(iter_,'t'), p+3] = 'Add X' + str(added_var[-1]+1)
            return True, df