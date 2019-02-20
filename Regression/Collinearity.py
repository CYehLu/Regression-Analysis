import numpy as np


class ExtraSS:
    def __init__(self, regobj):
        if regobj.__class__.__name__ != 'LinearRegression':
            raise TypeError(f"'regobj' should be 'LinearRegression', not '{regobj.__class__.__name__}'")
        else:
            self.regobj = regobj
            
    def test(self):
        print(self.regobj.coefficient_)