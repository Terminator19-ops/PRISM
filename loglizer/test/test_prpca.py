#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Validation suite for PRPCA-CS (Projected Robust Principal Component Analysis - Column Space).
Verifies:
1. Training completes without errors.
2. Learned low-rank matrix has valid dimensions.
3. Projection works for unseen samples of varying batch sizes (including N_test = 1).
4. Test predictions have the exact correct number of entries.
5. Residual scores are finite (no NaNs, no Infs).
6. Predictions contain strictly binary values {0, 1}.
7. Reproducibility under identical random seeds.
8. Test set isolation (unseen during fitting and projection matrix construction).
9. Synthetic low-rank matrix with known ground-truth anomalies.
"""

import os
import sys
import unittest
import numpy as np

# Ensure project root is on sys.path
test_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(test_dir, '..'))
sys.path.insert(0, project_dir)

from loglizer.models.RPCA import R_pca


class TestPRPCACS(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.n_train = 80
        self.n_features = 12
        self.rank_true = 3

        # Construct a known low-rank matrix L_true = A @ B
        A_train = np.random.randn(self.n_train, self.rank_true)
        B = np.random.randn(self.rank_true, self.n_features)
        self.L_true = np.dot(A_train, B)

        # Inject sparse anomalies in training data
        self.S_train_true = np.zeros_like(self.L_true)
        self.S_train_true[5, 2] = 5.0
        self.S_train_true[15, 7] = 6.0
        self.X_train = self.L_true + self.S_train_true

        # Unseen test data: 20 normal, 5 anomalous
        self.n_test_normal = 20
        self.n_test_anomaly = 5
        A_test_normal = np.random.randn(self.n_test_normal, self.rank_true)
        X_test_normal = np.dot(A_test_normal, B)

        A_test_anomaly = np.random.randn(self.n_test_anomaly, self.rank_true)
        X_test_anomaly = np.dot(A_test_anomaly, B)
        # Inject large deviations in anomaly samples
        X_test_anomaly[:, 3] += 8.0

        self.X_test = np.vstack([X_test_normal, X_test_anomaly])
        self.y_test = np.array([0] * self.n_test_normal + [1] * self.n_test_anomaly)

    def test_1_and_2_training_completion_and_dimensions(self):
        """1. Training completes without errors; 2. Low-rank matrix has valid dimensions."""
        model = R_pca(self.X_train, threshold=2.0, n_components=0.95)
        L, S = model.fit2(max_iter=300)

        self.assertIsNotNone(L)
        self.assertIsNotNone(S)
        self.assertEqual(L.shape, self.X_train.shape)
        self.assertEqual(S.shape, self.X_train.shape)
        self.assertIsNotNone(model.proj_C)
        self.assertEqual(model.proj_C.shape, (self.n_features, self.n_features))

    def test_3_projection_unseen_samples_varying_sizes(self):
        """3. Projection works for unseen samples of various batch sizes (N=1, N=5, N=25)."""
        model = R_pca(self.X_train, threshold=2.0, n_components=0.95)
        model.fit2(max_iter=300)

        # Test single sample (1, n_features)
        single_sample = self.X_test[0:1, :]
        pred_single = model.predict(single_sample)
        self.assertEqual(pred_single.shape, (1,))

        # Test arbitrary batch (7, n_features)
        batch_sample = self.X_test[0:7, :]
        pred_batch = model.predict(batch_sample)
        self.assertEqual(pred_batch.shape, (7,))

        # Test full test set (25, n_features)
        pred_full = model.predict(self.X_test)
        self.assertEqual(pred_full.shape, (self.X_test.shape[0],))

    def test_4_and_6_prediction_entries_and_binary_values(self):
        """4. Test predictions have correct number of entries; 6. Values are strictly {0, 1}."""
        model = R_pca(self.X_train, threshold=2.0, n_components=0.95)
        model.fit2(max_iter=300)

        y_pred = model.predict(self.X_test)
        self.assertEqual(len(y_pred), len(self.y_test))
        unique_vals = set(np.unique(y_pred))
        self.assertTrue(unique_vals.issubset({0, 1}))

    def test_5_residuals_finite(self):
        """5. Residual scores are finite (no NaNs, no Infs)."""
        model = R_pca(self.X_train, threshold=2.0, n_components=0.95)
        model.fit2(max_iter=300)

        L_proj = np.dot(self.X_test, model.proj_C)
        S_proj = self.X_test - L_proj

        self.assertTrue(np.all(np.isfinite(L_proj)))
        self.assertTrue(np.all(np.isfinite(S_proj)))

    def test_7_reproducibility(self):
        """7. Repeated runs with same input data and parameters produce identical results."""
        model1 = R_pca(self.X_train, threshold=2.0, n_components=0.95)
        L1, S1 = model1.fit2(max_iter=300)
        pred1 = model1.predict(self.X_test)

        model2 = R_pca(self.X_train, threshold=2.0, n_components=0.95)
        L2, S2 = model2.fit2(max_iter=300)
        pred2 = model2.predict(self.X_test)

        np.testing.assert_allclose(L1, L2, rtol=1e-5)
        np.testing.assert_allclose(S1, S2, rtol=1e-5)
        np.testing.assert_array_equal(pred1, pred2)

    def test_8_test_set_isolation(self):
        """8. Test set is not referenced or used during training/subspace extraction."""
        model = R_pca(self.X_train, threshold=2.0, n_components=0.95)
        model.fit2(max_iter=300)

        # Assert projection matrix only depends on self.L from training
        U_tr, s_tr, Vh_tr = np.linalg.svd(model.L, full_matrices=False)
        k = model.k_components
        Vk = Vh_tr.T[:, :k]
        expected_proj = np.dot(Vk, Vk.T)

        np.testing.assert_allclose(model.proj_C, expected_proj, rtol=1e-6)

    def test_9_synthetic_anomaly_detection(self):
        """Verify model accurately separates normal samples from injected anomalous deviations."""
        model = R_pca(self.X_train, threshold=2.0, n_components=0.95)
        model.fit2(max_iter=300)

        y_pred = model.predict(self.X_test)
        metrics_dict = model.calc_TP_FP_rate(self.y_test, y_pred)

        self.assertEqual(metrics_dict['TP'], self.n_test_anomaly)
        self.assertEqual(metrics_dict['FP'], 0)
        self.assertEqual(metrics_dict['Precision'], 1.0)
        self.assertEqual(metrics_dict['Recall'], 1.0)


if __name__ == '__main__':
    unittest.main()
