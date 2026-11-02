#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_and_save_model.py
=======================
Trains the PRPCA-CS baseline model on the OpenStack benchmark training set
and exports the serialized artifacts (feature extractor, projection matrix,
subspace rank, threshold) for fast inference in the FastAPI service.
"""

import os
import sys
import time
import joblib
import numpy as np

# Ensure loglizer is importable
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'loglizer'))

from loglizer import dataloader, preprocessing
from loglizer.models.RPCA import R_pca

DATA_DIR = os.path.join(PROJECT_ROOT, 'loglizer', 'data', 'OpenStack')
STRUCT_LOG = os.path.join(DATA_DIR, '20k+4k destroy+10k undefined +4k dhcp.log_structured.csv')
LABEL_FILE = os.path.join(DATA_DIR, '20k+4k destroy + 10k undefiend+4k dhcp_label.csv')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'prpca_model.joblib')

def train_and_export():
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("Loading OpenStack dataset...")
    t0 = time.time()
    (x_train, y_train), (x_test, y_test), df = dataloader.load_OpenStack(
        STRUCT_LOG,
        label_file=LABEL_FILE,
        window='session',
        train_ratio=0.5,
        random_state=20
    )
    y_train = np.asarray(y_train, dtype=int)
    y_test = np.asarray(y_test, dtype=int)

    print("Fitting FeatureExtractor (TF-IDF, 28 events)...")
    feature_extractor = preprocessing.FeatureExtractor()
    X_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf')
    X_test = feature_extractor.transform(x_test)

    print("Fitting PRPCA-CS (max_iter=600, n_components=0.95, threshold=0.8)...")
    t_fit = time.time()
    model = R_pca(X_train, threshold=0.8, n_components=0.95)
    L, S = model.fit2(max_iter=600)
    fit_duration = time.time() - t_fit
    print(f"Model fit completed in {fit_duration:.2f}s (k_components={model.k_components})")

    # Evaluate on test set for verification
    y_pred_test = model.predict(X_test)
    test_metrics = model.calc_TP_FP_rate(y_test, y_pred_test)
    print(f"Test verification metrics: TP={test_metrics['TP']}, TN={test_metrics['TN']}, "
          f"FP={test_metrics['FP']}, FN={test_metrics['FN']}, "
          f"Precision={test_metrics['Precision']:.4f}, Recall={test_metrics['Recall']:.4f}, "
          f"F1={test_metrics['F1']:.4f}, Accuracy={test_metrics['Accuracy']:.4f}")

    # Build artifact bundle
    artifact = {
        'model_name': 'PRPCA-CS',
        'proj_C': model.proj_C,
        'k_components': int(model.k_components),
        'threshold': float(model.threshold),
        'n_components': float(model.n_components),
        'events': list(feature_extractor.events),
        'idf_vec': feature_extractor.idf_vec,
        'term_weighting': feature_extractor.term_weighting,
        'normalization': feature_extractor.normalization,
        'feature_extractor': feature_extractor,
        'train_samples': int(len(y_train)),
        'num_features': int(model.proj_C.shape[0]),
        'verification_metrics': test_metrics
    }

    joblib.dump(artifact, MODEL_PATH)
    file_size_kb = os.path.getsize(MODEL_PATH) / 1024
    print(f"\nModel successfully saved to {MODEL_PATH} ({file_size_kb:.2f} KB)")
    print(f"Total training pipeline time: {time.time() - t0:.2f}s")
    return artifact

if __name__ == '__main__':
    train_and_export()
