#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os

# Get the project root directory
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)
sys.path.insert(0, project_root)

from loglizer.models.RPCA import R_pca
from loglizer import dataloader, preprocessing

demo_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(demo_dir)
struct_log = os.path.join(
    project_dir, 'data', 'OpenStack',
    '20k+4k destroy+10k undefined +4k dhcp.log_structured.csv'
)

label_file = os.path.join(
    project_dir, 'data', 'OpenStack',
    '20k+4k destroy + 10k undefiend+4k dhcp_label.csv'
)

print("Current working directory:", os.getcwd())
print("Structured log exists:", os.path.exists(struct_log))
print("Label file exists:", os.path.exists(label_file))


def print_detailed_metrics(title, metrics_dict):
    print(f"\n--- Detailed Metrics: {title} ---")
    print(f"True Positives  (TP) : {metrics_dict['TP']}")
    print(f"True Negatives  (TN) : {metrics_dict['TN']}")
    print(f"False Positives (FP) : {metrics_dict['FP']}")
    print(f"False Negatives (FN) : {metrics_dict['FN']}")
    print(f"Precision            : {metrics_dict['Precision']:.4f}")
    print(f"Recall               : {metrics_dict['Recall']:.4f}")
    print(f"F1-score             : {metrics_dict['F1']:.4f}")
    print(f"Accuracy             : {metrics_dict['Accuracy']:.5f}")
    print(f"False Positive Rate  : {metrics_dict['FPR']:.4f}")


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test), DataFrame = dataloader.load_OpenStack(
        struct_log,
        label_file=label_file,
        window='session',
        train_ratio=0.5,
        random_state=20
    )

    feature_extractor = preprocessing.FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf')
    x_test = feature_extractor.transform(x_test)

    model = R_pca(x_train, threshold=0.8, n_components=0.95)
    L, S = model.fit2(max_iter=600, iter_print=100)

    print('Train validation (RPCA Sparse Decomposition):')
    precision_tr, recall_tr, f1_tr, accuracy_tr, y_pred_tr = model.evaluate1(x_train, y_train)
    m_tr = model.calc_TP_FP_rate(y_train, y_pred_tr)
    print_detailed_metrics('Train Set (RPCA Decomposition)', m_tr)

    print('\nTest validation (PRPCA-CS Subspace Projection):')
    precision_te, recall_te, f1_te, accuracy_te, y_pred_te = model.evaluate2(x_test, y_test)
    m_te = model.calc_TP_FP_rate(y_test, y_pred_te)
    print_detailed_metrics('Test Set (PRPCA-CS Projection)', m_te)
