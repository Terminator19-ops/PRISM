from __future__ import division, print_function

import numpy as np
import pandas as pd
from loglizer.utils import metrics
from sklearn import preprocessing


class R_pca:

    def __init__(self, D, mu=None, lmbda=None, threshold=None, n_components=0.95):
        """
        Robust Principal Component Analysis (RPCA) via Inexact ALM / PCP,
        extended with Projection onto Column Space (PRPCA-CS) for out-of-sample prediction.

        Parameters:
        -----------
        D: ndarray of shape (n_samples, n_features)
            Training feature matrix (session-by-event).
        mu: float, optional
            Augmented Lagrange Multiplier step size.
        lmbda: float, optional
            Sparse matrix penalty weight.
        threshold: float, optional
            Threshold applied to sparse residual to classify anomalies.
        n_components: float or int, optional
            Subspace rank selection parameter (default 0.95 for 95% explained variance).
        """
        self.D = D
        self.S = np.zeros(self.D.shape)
        self.Y = np.zeros(self.D.shape)
        self.M = np.zeros(self.D.shape)
        self.threshold = threshold
        self.var = []
        self.n_components = n_components
        self.L = None
        self.proj_C = None
        self.k_components = None

        if mu:
            self.mu = mu
        else:
            self.mu = np.prod(self.D.shape) / (4.0 * np.linalg.norm(self.D, ord=1))

        self.mu_inv = 1.0 / self.mu

        if lmbda:
            self.lmbda = lmbda
        else:
            self.lmbda = 1.0 / np.sqrt(np.max(self.D.shape))

    @staticmethod
    def frobenius_norm(M):
        return np.linalg.norm(M, ord='fro')

    @staticmethod
    def shrink(M, tau):
        return np.sign(M) * np.maximum((np.abs(M) - tau), np.zeros(M.shape))

    def svd_threshold(self, M, tau):
        U, S, V = np.linalg.svd(M, full_matrices=False)
        self.M = M
        return np.dot(U, np.dot(np.diag(self.shrink(S, tau)), V))

    def fit2(self, tol=None, max_iter=1000, iter_print=100):
        """
        Fit the RPCA model via Principal Component Pursuit (Inexact ALM).
        Decomposes D = L + S into:
            L: low-rank matrix representing normal routine executions
            S: sparse matrix representing anomalous deviations
        """
        iter_count = 0
        err = np.inf
        Sk = self.S
        Yk = self.Y
        Lk = np.zeros(self.D.shape)

        if tol:
            _tol = tol
        else:
            _tol = 1E-7 * self.frobenius_norm(self.D)

        while (err > _tol) and iter_count < max_iter:
            Lk = self.svd_threshold(self.D - Sk + self.mu_inv * Yk, self.mu_inv)
            Sk = self.shrink(
                self.D - Lk + (self.mu_inv * Yk), self.mu_inv * self.lmbda)
            Yk = Yk + self.mu * (self.D - Lk - Sk)
            err = self.frobenius_norm(self.D - Lk - Sk)
            iter_count += 1

        self.L = Lk
        self.S = Sk

        # Compute the projection matrix onto the normal subspace in feature space
        self._compute_projection_matrix()

        return Lk, Sk

    def _compute_projection_matrix(self):
        r"""
        Extract the normal feature subspace from the learned low-rank matrix L.

        Mathematical Formulation:
        -------------------------
        Matrix orientation:
            self.D, self.L have shape (n_samples, n_features) = (n, m).
            Each row x_i is a session observation in R^m (feature space).

        SVD of L:
            L = U \Sigma V^T
        where:
            U is (n, m): left singular vectors (session space R^n)
            \Sigma is diagonal matrix of singular values
            V is (m, m): right singular vectors, whose columns form an
            orthonormal basis for the row space of L (or column space of L^T),
            which corresponds directly to the feature space R^m.

        Subspace rank selection (k):
            Following the paper, n_components (default 0.95) selects the smallest
            k such that the cumulative explained singular spectrum ratio >= n_components.
            To prevent full-dimensional projections (which would cause zero residuals),
            k is capped at max(1, min(k, m - 1)).

        Projection matrix:
            V_k = V[:, :k]  (shape: m-by-k)
            P = V_k V_k^T   (shape: m-by-m, symmetric and idempotent)

        For any sample batch X in R^(N x m):
            L_proj = X @ P        (projection onto normal subspace)
            S_proj = X - L_proj   (residual in anomaly subspace)
        """
        if self.L is None:
            raise ValueError("Model must be fitted with fit2() before computing projection matrix.")

        U, s, Vh = np.linalg.svd(self.L, full_matrices=False)
        V = Vh.T  # shape: (n_features, min(n, n_features))

        total_variance = np.sum(s)
        if total_variance <= 0:
            k = 1
        elif self.n_components < 1.0:
            cum_var = np.cumsum(s) / total_variance
            k = int(np.searchsorted(cum_var, self.n_components) + 1)
        else:
            k = int(self.n_components)

        num_features = self.D.shape[1]
        k = max(1, min(k, num_features - 1 if num_features > 1 else 1))

        self.k_components = k
        Vk = V[:, :k]  # shape: (num_features, k)
        self.proj_C = np.dot(Vk, Vk.T)  # shape: (num_features, num_features)
        return self.proj_C

    def predict(self, X):
        """
        Predict anomaly labels for input feature matrix X using PRPCA-CS.

        Projects X onto the learned normal subspace and evaluates residuals against threshold.

        Parameters:
        -----------
        X: ndarray of shape (n_samples, n_features)

        Returns:
        --------
        y_pred: ndarray of shape (n_samples,) with binary values {0, 1}
        """
        if self.proj_C is None:
            self._compute_projection_matrix()

        if X.shape[1] != self.proj_C.shape[0]:
            raise ValueError(f"Feature dimension mismatch: X has {X.shape[1]} features, "
                             f"but model projection expects {self.proj_C.shape[0]} features.")

        # Project X onto the normal subspace
        L_proj = np.dot(X, self.proj_C)
        S_proj = X - L_proj

        # Decision rule: flag anomaly if any event residual exceeds threshold
        y_pred = np.zeros(X.shape[0], dtype=int)
        for i in range(X.shape[0]):
            TF = S_proj[i] > self.threshold
            if TF.any():
                y_pred[i] = 1

        return y_pred

    def evaluate1(self, X, y_true):
        """
        Evaluate performance on training data using the RPCA decomposition sparse matrix S.
        """
        print('====== Evaluation summary (RPCA Sparse Decomposition) ======')
        y_pred = np.zeros(X.shape[0], dtype=int)
        S = self.S

        for i in range(X.shape[0]):
            TF = S[i] > self.threshold
            if TF.any():
                y_pred[i] = 1

        precision, recall, f1, accuracy = metrics(y_pred, y_true)
        print('Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}, Accuracy: {:.5f} \n'.format(
            precision, recall, f1, accuracy))
        return precision, recall, f1, accuracy, y_pred

    def evaluate2(self, X, y_true):
        """
        Evaluate performance on test data using PRPCA-CS projection onto the normal subspace.
        """
        print('====== Evaluation summary (PRPCA-CS Subspace Projection) ======')
        y_pred = self.predict(X)
        precision, recall, f1, accuracy = metrics(y_pred, y_true)
        print('Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}, Accuracy: {:.5f} \n'.format(
            precision, recall, f1, accuracy))
        return precision, recall, f1, accuracy, y_pred

    def calc_TP_FP_rate(self, y_true, y_pred):
        """
        Calculate confusion matrix and classification metrics.

        Returns:
        --------
        dict containing TP, TN, FP, FN, Precision, Recall, F1, Accuracy, FPR.
        """
        y_true = np.asarray(y_true).astype(int)
        y_pred = np.asarray(y_pred).astype(int)

        TP = int(np.sum((y_true == 1) & (y_pred == 1)))
        TN = int(np.sum((y_true == 0) & (y_pred == 0)))
        FP = int(np.sum((y_true == 0) & (y_pred == 1)))
        FN = int(np.sum((y_true == 1) & (y_pred == 0)))

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (TP + TN) / len(y_true) if len(y_true) > 0 else 0.0
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0

        return {
            'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
            'Precision': precision, 'Recall': recall,
            'F1': f1, 'Accuracy': accuracy, 'FPR': fpr
        }
