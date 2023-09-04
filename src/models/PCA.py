import numpy as np

class PCA:
    def __init__(self, n_components=2, preprocess="standardize", random_state=42):
        """
        Args:
            n_components (int): number of components to keep
            random_state (int): random seed
        """
        self.n_components = n_components
        self.random_state = random_state
        self.components = np.array([])
        self.eigenvectors = None
        self.explained_variance = None
        self.explained_variance_ratio = None
        self.diag = np.array([])
        self.unitary_matrix = np.array([])
        self.X_mean = np.array([])
        self.X_std = np.array([])
        self.preprocess = preprocess


    def fit(self, X):
        """
        Description:
        1. Standardize the data
        2. Compute the covariance matrix
        3. Compute the eigenvectors and eigenvalues of the covariance matrix
        4. Sort the eigenvectors by eigenvalues in decreasing order
        5. Select the top n components

        Args:
            X (np.ndarray): data matrix

        Returns:
            np.ndarray: top n components
        """
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array.")

        self.X_mean = X.mean(axis=0)

        if self.preprocess == "standardize":
            self.X_std = X.std(axis=0)
            X = self._standardize(X)
        elif self.preprocess == "center":
            X = self._center(X)

        # NOTE: compute using the covariance matrix
        # covariance_matrix = np.cov(X.T)
        # eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        # self.eigenvectors = eigenvectors
        # self.components = eigenvectors[:, :self.n_components].T
        # self.explained_variance_ratio = eigenvalues / eigenvalues.sum()

        # NOTE: compute using the singular value decomposition
        U, s, eigenvectors = np.linalg.svd(X, full_matrices=False)
        self.eigenvalues = (s**2) / (X.shape[0] - 1)

        self.diag = s
        self.unitary_matrix = U

        self.eigenvectors = eigenvectors
        self.components = eigenvectors[:self.n_components]
        self.explained_variance_ratio = self.eigenvalues / self.eigenvalues.sum()

        return self.components


    def transform(self, X):
        """
        Project the data onto the selected components.

        Args:
            X (np.ndarray): data matrix

        Returns:
            np.ndarray: projected data matrix
        """
        if not self.components.any():
            raise ValueError("PCA has not been fitted yet.")

        if self.preprocess == "standardize":
            X = self._standardize(X)
        elif self.preprocess == "center":
            X = self._center(X)

        return X.dot(self.components.T)


    def _center(self, X):
        if not self.X_mean.any():
            self.X_mean = X.mean(axis=0)

        return X - self.X_mean


    def _standardize(self, X):
        if not self.X_mean.any():
            self.X_mean = X.mean(axis=0)

        if not self.X_std.any():
            self.X_std = X.std(axis=0)

        return (X - self.X_mean) / self.X_std
