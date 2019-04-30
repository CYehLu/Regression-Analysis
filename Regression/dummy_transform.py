import numpy as np


def dummy(xcol):
    """
    Transform a column vector to dummy variable matrix.
    
    EX:
    >>> xcol = np.array([[1],
                         [2],
                         [1],
                         [2],
                         [3]])
    >>> dummy(xcol)
    np.array([[0, 0],
             [1, 0],
             [0, 0],
             [1, 0],
             [0, 1]])
    """
    types = np.unique(xcol)
    n = len(types)
    
    dummy_matrix = np.zeros((xcol.size, n), dtype=int)
    for i in range(n):
        dummy_matrix[(xcol == types[i]).ravel(),i] = 1
        
    return dummy_matrix[:,1:]

def dummy_expand(X, dummy_idx):
    """
    Expand column to dummy variable form by given expand index.
    
    Parameter:
    ---------
    X: 2-d array
    dummy_idx: list. The index of column which will be expanded into
               dummy form.
               
    Return:
    ------
    Expanded X
    

    EX: 
    >>> X = np.array([[1, 2, 1],
                      [0, 3, 1],
                      [1, 1, 2],
                      [2, 4, 0]])
    >>> dummy_idx = [0, 2]
    
    >>> dummy_expand(X, dummy_idx)   
    X = np.array([[1, 0, 2, 1, 0],
                  [0, 0, 3, 1, 0],
                  [1, 0, 1, 0, 1],
                  [0, 1, 4, 0, 0]])
    # The 0 and 2 column in X has been expand into dummy form.
    """
    split_X = np.hsplit(X, range(1, X.shape[1]))
    for idx in dummy_idx:
        split_X[idx] = dummy(split_X[idx])
        
    return np.concatenate(split_X, axis=1)