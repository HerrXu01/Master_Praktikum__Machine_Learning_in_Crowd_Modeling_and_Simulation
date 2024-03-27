import numpy as np
import matplotlib.pyplot as plt

def coordinate_extractor(file_path):
    
    """
    
    Extracts and returns a numpy 2d array of x and y values from the given file.
    
    Args:
        
        - file_path (str): path to the required txt file.
    
    Returns:
    
        - np.ndarray: A matrix of coordinates.
        
    """
    # Load data from the text file using numpy
    data = np.loadtxt(file_path)

    # Slicing 'data' to keep only first 2 columns
    matrix = data[:, :2]
    
    return matrix

def mean_squared_error(y_true, y_pred):
    """
    
    Calculate the mean squared error between two 2D arrays.

    Args:
    
        - y_true (numpy.ndarray): The true values (ground truth).
        - y_pred (numpy.ndarray): The predicted values.

    Returns:
        - float: The mean squared error.
    
    """
    mse = np.mean((y_true - y_pred)**2)
    return mse

def phase_portrait(A, title_suffix):
    """
    
    Creates a phase portrait matrix based on the given linear operator.

    Args:
        
        - A (numpy.ndarray): Linear operator.
        - title_suffix (str): Suffix for the plot title.
        - display (bool): Whether to display the plot.

    Returns:
        
        - None
        
    """
    w = 10  # width
    Y, X = np.mgrid[-w:w:100j, -w:w:100j]
    #eigenvalues = np.linalg.eigvals(A)
    #print("Eigenvalues of A: ", eigenvalues)
    # linear vector field A*x
    UV = A @ np.row_stack([X.ravel(), Y.ravel()])
    U = UV[0, :].reshape(X.shape)
    V = UV[1, :].reshape(X.shape)
    fig = plt.figure(figsize=(10, 10))
    plt.streamplot(X, Y, U, V, density=1.0)
    plt.title(title_suffix)
    plt.show()