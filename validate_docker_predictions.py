#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
validate_docker_predictions.py
==============================
Validates that Dockerized FastAPI REST service predictions exactly match
the local Python PRPCA-CS inference pipeline on the same dataset.
"""

import os
import sys
import json
import urllib.request
import urllib.error
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'loglizer'))

from app.inference import PRPCAInferenceEngine

SAMPLE_CSV = os.path.join(PROJECT_ROOT, 'tests', 'data', 'sample_valid_sessions.csv')
DOCKER_API_URL = "http://localhost:8000/predict"

def validate_parity(threshold=0.8):
    print("=" * 70)
    print("PRPCA-CS VALIDATION: Direct Python vs Docker Container API")
    print(f"Testing with threshold: {threshold}")
    print("=" * 70)

    # 1. Run inference directly in Python
    print("\n[1] Running direct Python PRPCA-CS inference...")
    engine = PRPCAInferenceEngine()
    with open(SAMPLE_CSV, 'rb') as f:
        file_bytes = f.read()

    py_result = engine.predict_from_file(file_bytes, threshold_override=threshold)
    print(f"  Direct Python results:")
    print(f"    Total Sessions     : {py_result['total_sessions']}")
    print(f"    Normal Sessions    : {py_result['normal_sessions']}")
    print(f"    Anomalous Sessions : {py_result['anomalous_sessions']}")
    print(f"    Anomaly Rate       : {py_result['anomaly_rate']:.4f}")
    print(f"    Predictions        : {py_result['predictions']}")

    # 2. Run inference via Docker REST API
    print("\n[2] Querying Docker API endpoint...")
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    filename = os.path.basename(SAMPLE_CSV)

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: text/csv\r\n\r\n"
    ).encode('utf-8') + file_bytes + f"\r\n--{boundary}--\r\n".encode('utf-8')

    url = f"{DOCKER_API_URL}?threshold={threshold}"
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req) as response:
            docker_response_bytes = response.read()
            docker_result = json.loads(docker_response_bytes.decode('utf-8'))
    except Exception as e:
        print(f"ERROR connecting to Docker API at {url}: {e}")
        return False

    print(f"  Docker API results:")
    print(f"    Total Sessions     : {docker_result['total_sessions']}")
    print(f"    Normal Sessions    : {docker_result['normal_sessions']}")
    print(f"    Anomalous Sessions : {docker_result['anomalous_sessions']}")
    print(f"    Anomaly Rate       : {docker_result['anomaly_rate']:.4f}")
    print(f"    Predictions        : {docker_result['predictions']}")

    # 3. Assertions and Parity Checks
    print("\n[3] Running parity assertions...")
    assert py_result['total_sessions'] == docker_result['total_sessions'], "Session count mismatch!"
    assert py_result['normal_sessions'] == docker_result['normal_sessions'], "Normal count mismatch!"
    assert py_result['anomalous_sessions'] == docker_result['anomalous_sessions'], "Anomaly count mismatch!"
    assert np.isclose(py_result['anomaly_rate'], docker_result['anomaly_rate']), "Anomaly rate mismatch!"
    assert py_result['predictions'] == docker_result['predictions'], "Predictions array mismatch!"

    for py_s, doc_s in zip(py_result['session_details'], docker_result['session_details']):
        assert py_s['instance_id'] == doc_s['instance_id'], "Instance ID mismatch!"
        assert py_s['prediction'] == doc_s['prediction'], "Session prediction mismatch!"
        assert np.isclose(py_s['max_residual'], doc_s['max_residual'], atol=1e-5), "Residual score mismatch!"

    print("  ✓ Session counts match perfectly")
    print("  ✓ Normal/Anomaly distribution matches perfectly")
    print("  ✓ Per-session predictions match 100%")
    print("  ✓ Max residual scores match to numerical precision")
    print("  >> VALIDATION PASSED: 100% PARITY CONFIRMED <<\n")
    return True

if __name__ == '__main__':
    # Validate with default baseline threshold 0.8
    ok1 = validate_parity(threshold=0.8)
    # Validate with lower threshold 0.2 to test anomaly triggering parity
    ok2 = validate_parity(threshold=0.2)
    if ok1 and ok2:
        print("ALL TESTS PASSED SUCCESSFULLY.")
        sys.exit(0)
    else:
        sys.exit(1)

