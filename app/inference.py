# -*- coding: utf-8 -*-
"""
app/inference.py
================
Inference engine for PRPCA-CS OpenStack log anomaly detection.
Loads serialized model artifacts and performs out-of-sample projection
onto the learned normal subspace.
"""

import io
import os
import re
import sys
from collections import OrderedDict
from typing import Dict, Any, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

# Add project root and loglizer to sys.path so unpickling works
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
LOGLIZER_DIR = os.path.join(PROJECT_ROOT, 'loglizer')
if LOGLIZER_DIR not in sys.path:
    sys.path.insert(0, LOGLIZER_DIR)

from loglizer.models.RPCA import R_pca


class InvalidLogFileError(ValueError):
    """Raised when an uploaded log file has invalid structure or missing columns."""
    pass


class PRPCAInferenceEngine:
    """
    Production inference engine wrapping the learned PRPCA-CS subspace model.
    """

    SUPPORTED_FORMAT_MSG = (
        "Supported format: Structured CSV log file containing at least two columns: "
        "'Content' (with session identifiers matching 'instance: <uuid>') and 'EventId' "
        "(parsed template ID, e.g., '8d02d411'). Raw unstructured log files must first be "
        "processed through a log parser such as IPLoM or Drain before submitting for anomaly detection."
    )

    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            model_path = os.path.join(PROJECT_ROOT, 'model', 'prpca_model.joblib')

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model artifact not found at {model_path}. "
                "Please run 'python train_and_save_model.py' first."
            )

        self.model_path = model_path
        self._load_artifacts()

    def _load_artifacts(self):
        print(f"[InferenceEngine] Loading artifacts from {self.model_path}...")
        artifact = joblib.load(self.model_path)

        self.model_name = artifact.get('model_name', 'PRPCA-CS')
        self.proj_C = artifact['proj_C']  # shape: (n_features, n_features)
        self.k_components = artifact['k_components']
        self.threshold = artifact['threshold']
        self.n_components = artifact['n_components']
        self.feature_extractor = artifact['feature_extractor']
        self.events = artifact['events']
        self.num_features = artifact['num_features']
        self.train_samples = artifact.get('train_samples', 0)
        print(f"[InferenceEngine] Artifacts loaded successfully. Features: {self.num_features}, "
              f"k={self.k_components}, default threshold={self.threshold}")

    def validate_and_parse_csv(self, file_content: bytes) -> Tuple[List[str], np.ndarray]:
        """
        Validates the uploaded file buffer and extracts OpenStack VM sessions.

        Returns
        -------
        instance_ids: List of instance ID strings
        event_sequences: ndarray of event ID sequences
        """
        if not file_content or len(file_content.strip()) == 0:
            raise InvalidLogFileError("The uploaded file is empty.")

        # Attempt to parse as CSV
        try:
            df = pd.read_csv(
                io.BytesIO(file_content),
                engine='c',
                na_filter=False
            )
        except Exception as e:
            # Check if this looks like a raw log file instead of CSV
            preview = file_content[:500].decode('utf-8', errors='ignore')
            if ('nova.' in preview and 'HTTP' in preview) or (preview.startswith('INFO ') or preview.startswith('DEBUG ')):
                raise InvalidLogFileError(
                    f"Uploaded file appears to be an unparsed raw OpenStack log file. {self.SUPPORTED_FORMAT_MSG}"
                )
            raise InvalidLogFileError(
                f"Failed to parse uploaded file as CSV: {str(e)}. {self.SUPPORTED_FORMAT_MSG}"
            )

        # Validate required columns
        required_cols = {'Content', 'EventId'}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise InvalidLogFileError(
                f"Missing required columns in CSV: {sorted(list(missing_cols))}. "
                f"Available columns: {list(df.columns)}. {self.SUPPORTED_FORMAT_MSG}"
            )

        if len(df) == 0:
            raise InvalidLogFileError("The uploaded CSV has headers but contains 0 data rows.")

        # Extract sessions via regex matching on Content
        instance_pattern = re.compile(r'(instance: \w*-\w*-\w*-\w*-\w*)')
        data_dict = OrderedDict()

        for idx, row in df.iterrows():
            content = str(row['Content'])
            event_id = str(row['EventId'])
            blk_ids = instance_pattern.findall(content)
            for blk_id in set(blk_ids):
                if blk_id not in data_dict:
                    data_dict[blk_id] = []
                data_dict[blk_id].append(event_id)

        if len(data_dict) == 0:
            raise InvalidLogFileError(
                "No OpenStack VM sessions found in the uploaded file. The 'Content' column "
                "must contain VM instance references matching pattern 'instance: <uuid>'. "
                f"{self.SUPPORTED_FORMAT_MSG}"
            )

        instance_ids = list(data_dict.keys())
        event_sequences = np.array(list(data_dict.values()), dtype=object)

        return instance_ids, event_sequences

    def predict_from_file(
        self,
        file_content: bytes,
        threshold_override: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end PRPCA-CS inference on an uploaded structured log CSV.
        """
        instance_ids, event_sequences = self.validate_and_parse_csv(file_content)

        # Transform using the learned FeatureExtractor
        X_test = self.feature_extractor.transform(event_sequences)

        if X_test.shape[1] != self.proj_C.shape[0]:
            raise ValueError(
                f"Feature dimension mismatch: transformed data has {X_test.shape[1]} features, "
                f"but model projection expects {self.proj_C.shape[0]} features."
            )

        # Apply PRPCA-CS projection onto normal subspace
        L_proj = np.dot(X_test, self.proj_C)
        S_proj = X_test - L_proj

        active_threshold = threshold_override if threshold_override is not None else self.threshold

        # Compute per-session anomaly decisions and maximum residual
        n_sessions = X_test.shape[0]
        predictions = []
        session_details = []
        anomalous_count = 0

        for i in range(n_sessions):
            residuals = S_proj[i]
            max_res = float(np.max(residuals))
            is_anomaly = 1 if max_res > active_threshold else 0

            if is_anomaly == 1:
                anomalous_count += 1

            predictions.append(is_anomaly)
            session_details.append({
                'instance_id': instance_ids[i],
                'prediction': is_anomaly,
                'status': 'Anomaly' if is_anomaly == 1 else 'Normal',
                'max_residual': round(max_res, 6)
            })

        normal_count = n_sessions - anomalous_count
        anomaly_rate = round(anomalous_count / n_sessions, 4) if n_sessions > 0 else 0.0

        return {
            'total_sessions': n_sessions,
            'normal_sessions': normal_count,
            'anomalous_sessions': anomalous_count,
            'anomaly_rate': anomaly_rate,
            'threshold_used': round(float(active_threshold), 4),
            'predictions': predictions,
            'session_details': session_details,
            'model_info': {
                'model_name': self.model_name,
                'k_components': self.k_components,
                'num_features': self.num_features,
                'default_threshold': self.threshold
            }
        }
