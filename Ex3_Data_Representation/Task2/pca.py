import numpy as np

class PCA():
    """
    A class for Principal Components Analysis, with several related methods implemented.
    """

    def __init__(self, X):
        """
        Args:
            X: the input data points. 2-d numpy array with shape (N, d).
               N for N points, d is the dimensionality of each data point. 
        """
        self.X = X
        self.s = None

    def svd(self, full_matrices=True):
        """
        Perform singular value decomposition to the data.

        Args:
            full_matrices: If True (default), u (N, N) and vh (d, d) are square matrices;
                            If False, u and vh have shapes (N, r) and (r, d), where r = min(N, d).
        
        returns:
            u: left singular vectors;
            s: singular values in descending order;
            vh: right singular values;
        """
        X_tilde = self.X - self.X.mean(axis=0)
        u, s, vh = np.linalg.svd(X_tilde, full_matrices=full_matrices)
        self.s = s

        return u, s, vh

    def reconstruct(self, components: int, full_matrices: bool):
        """
        Reconstruct the matrix with the given number of principal components.

        Args:
            components: int, the number of principal components;
            full_matrices: argument for the svd method.

        return:
            result: 2-d numpy array, the resulting matrix.
        """
        u, s, vh = self.svd(full_matrices)
        # Note: if using this formula to reconstruct the matrix, set the full_matrices to False!
        s_new = np.zeros(s.shape[0])
        s_new[:components] = s[:components]
        s_hat = np.diag(s_new)
        result = u @ s_hat @ vh + self.X.mean(axis=0)

        return result

    def energy_loss(self, p):
        """
        Compute at what number of the principal components is the “energy” lost smaller than p.

        Args:
            p: percentage of the energy loss.
        
        return:
            the least number of components needed.
        """
        percentage = np.cumsum(self.s) / np.sum(self.s)
        num = np.where(percentage > (1 - p))[0][0] + 1
        
        return num