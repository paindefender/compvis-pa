import numpy as np


def dot_product(vector1, vector2):
    """ Implement dot product of the two vectors.
    Args:
        vector1: numpy array of shape (n, 1)
        vector2: numpy array of shape (n, 1)

    Returns:
        out: scalar value
    """
    out = None
    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    n1, m1 = vector1.shape
    n2, m2 = vector2.shape
    assert n1 == n2
    assert m1 == 1
    assert m2 == 1
    out = np.dot(vector1.T, vector2)[0][0]
    ######################################
    #        END OF YOUR CODE            #
    ######################################

    return out

def matrix_mult(M, vector1, vector2,vector3):
    """ Implement (vector2.T * vector3) * (M * vector1.T)
    Args:
        M: numpy matrix of shape (m, n)
        vector1: numpy array of shape (1, n)
        vector2: numpy array of shape (n, 1)
        vector3: numpy array of shape (n, 1)

    Returns:
        out: numpy matrix of shape (m, 1)
    """
    out = None
    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    m, n = M.shape
    assert vector1.shape == (1, n)
    assert vector2.shape == (n, 1)
    assert vector2.shape == vector3.shape

    out = np.dot(vector2.T, vector3) * np.dot(M, vector1.T)
    ######################################
    #        END OF YOUR CODE            #
    ######################################

    return out

def svd(matrix):
    """ Implement Singular Value Decomposition
    Args:
        matrix: numpy matrix of shape (m, n)

    Returns:
        u: numpy array of shape (m, m)
        s: numpy array of shape (k)
        v: numpy array of shape (n, n)
    """
    u = None
    s = None
    v = None
    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    u, s, v = np.linalg.svd(matrix, full_matrices=True)
    ######################################
    #        END OF YOUR CODE            #
    ######################################

    return u, s, v

def get_singular_values(matrix, n):
    """ Return top n singular values of matrix
    Args:
        matrix: numpy matrix of shape (m, w)
        n: number of singular values to output
        
    Returns:
        singular_values: array of shape (n)
    """
    singular_values = None
    u, s, v = svd(matrix)
    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    assert n <= len(s)
    singular_values = s[:n]
    ######################################
    #        END OF YOUR CODE            #
    ######################################
    return singular_values

