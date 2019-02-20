import numpy as np

class ResidualAnalysis:
        #### 1. check regobj is a regression obj
        #### 2. check regobj.residual_ exists
    def __init__(self, regobj):
        if regobj.__class__.__name__ not in ['LinearRegression' or 'SimpleRegression']:
            raise TypeError("'regobj' should be 'LinearRegression' or 'SimpleRegression',"
                            + f"not '{regobj.__class__.__name__}'")
        else:
            self.regobj = regobj
        