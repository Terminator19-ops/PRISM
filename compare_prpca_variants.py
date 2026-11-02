"""
compare_prpca_variants.py
=========================
Compares three variants of PRPCA-CS for OpenStack log anomaly detection.

Research paper:
  "Anomaly detection on OpenStack logs based on an improved robust principal
   component analysis model and its projection onto column space"
  Parisa Sadat Kalaki et al. DOI: https://doi.org/10.1002/spe.3164

Experiments
-----------
1. PRPCA-CS with fixed threshold (baseline, threshold=0.8)
2. PRPCA-CS with adaptive thresholds (mean+2σ, mean+3σ, P95, P97, P99)
3. PRPCA-CS + event-transition bigram features + adaptive thresholds

All experiments use the same train/test split (random_state=20 / random_state=42).
Thresholds are calibrated on training normal samples ONLY — test labels are
never used during model fitting or threshold selection.
"""

from __future__ import print_function, division

import sys
import os
import time
import json
import csv
import re
import warnings
import copy
from collections import Counter, OrderedDict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'loglizer'))

from loglizer import dataloader, preprocessing
from loglizer.models.RPCA import R_pca

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
STRUCT_LOG = os.path.join(
    SCRIPT_DIR,
    'loglizer', 'data', 'OpenStack',
    '20k+4k destroy+10k undefined +4k dhcp.log_structured.csv'
)
LABEL_FILE = os.path.join(
    SCRIPT_DIR,
    'loglizer', 'data', 'OpenStack',
    '20k+4k destroy + 10k undefiend+4k dhcp_label.csv'
)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'comparison_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Fixed seed — must not change across experiments
# ---------------------------------------------------------------------------
RANDOM_STATE = 20  # data shuffle seed (dataloader)
FIXED_THRESHOLD = 0.8
N_COMPONENTS = 0.95
MAX_ITER = 600
MAX_ITER_EXP3 = 200  # Exp3 uses 138 features (5× larger SVD per iter); cap at 200


# ===========================================================================
# Helper: compute per-sample anomaly scores from projection residuals
# ===========================================================================
def compute_scores(X, proj_C):
    """Return per-sample max residual (shape: n_samples,)."""
    S_proj = X - np.dot(X, proj_C)
    assert np.all(np.isfinite(S_proj)), "Non-finite residuals detected!"
    return np.max(S_proj, axis=1)


def predict_from_scores(scores, threshold):
    """Binary prediction: 1 if max-residual > threshold, else 0."""
    y_pred = (scores > threshold).astype(int)
    assert set(np.unique(y_pred)).issubset({0, 1}), "Predictions must be binary."
    return y_pred


def confusion_metrics(y_true, y_pred):
    """Return dict of TP, TN, FP, FN, Precision, Recall, F1, Accuracy, FPR."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    TP = int(np.sum((y_true == 1) & (y_pred == 1)))
    TN = int(np.sum((y_true == 0) & (y_pred == 0)))
    FP = int(np.sum((y_true == 0) & (y_pred == 1)))
    FN = int(np.sum((y_true == 1) & (y_pred == 0)))
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc  = (TP + TN) / len(y_true) if len(y_true) > 0 else 0.0
    fpr  = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    return dict(TP=TP, TN=TN, FP=FP, FN=FN,
                Precision=prec, Recall=rec, F1=f1, Accuracy=acc, FPR=fpr)


# ===========================================================================
# Adaptive threshold calibration
# ===========================================================================
def calibrate_thresholds(train_scores_normal):
    """
    Compute adaptive threshold values using ONLY normal training scores.
    Returns a dict of {label: threshold_value}.
    """
    mu  = float(np.mean(train_scores_normal))
    std = float(np.std(train_scores_normal))
    thresholds = {
        'mean+2std': mu + 2 * std,
        'mean+3std': mu + 3 * std,
        'P95': float(np.percentile(train_scores_normal, 95)),
        'P97': float(np.percentile(train_scores_normal, 97)),
        'P99': float(np.percentile(train_scores_normal, 99)),
    }
    return thresholds


# ===========================================================================
# Event-transition bigram feature builder
# ===========================================================================
class BigramFeatureExtractor:
    """
    Builds event-transition bigram (consecutive event pairs) frequency features
    from ordered EventSequence lists.

    The vocabulary is built from TRAINING data only and applied to test data
    (unseen bigrams map to zero — no data leakage).

    Note on temporal features
    -------------------------
    The structured CSV has NO wall-clock timestamps (columns: LineId, LevelName,
    name, request_id, Content, EventId, EventTemplate, ParameterList).
    LineId is monotonically increasing and preserves log ordering within sessions,
    which is used here to extract ordered bigrams.

    The raw log format also contains no timestamps:
      LevelName  name  request_id  Content

    A subset of event templates contain embedded timing strings of the form
    "Took X.XX seconds to build/spawn", which are extracted as numeric features
    (spawn_time, build_time) where available; missing values are zero-filled.
    """

    # EventIds that contain "Took X.XX seconds" timing strings (from data inspection)
    SPAWN_EVENT = '41a2da91'   # "Took X.XX seconds to spawn the instance"
    BUILD_EVENT = '7824976e'   # "Took X.XX seconds to build the instance"
    EXTRA_EVENT = '36893f37'   # third timing event

    def __init__(self):
        self.bigram_vocab = None      # dict: bigram_tuple -> column_index
        self.n_bigrams = 0

    # ------------------------------------------------------------------
    # Internal: extract bigrams from a single EventSequence list
    # ------------------------------------------------------------------
    @staticmethod
    def _sequence_to_bigrams(seq):
        """[(e1, e2), (e2, e3), ...] from seq."""
        return [(seq[i], seq[i + 1]) for i in range(len(seq) - 1)]

    # ------------------------------------------------------------------
    # Build vocabulary from training sequences (call once on x_train)
    # ------------------------------------------------------------------
    def fit(self, x_train_sequences):
        bigram_counter = Counter()
        for seq in x_train_sequences:
            bigram_counter.update(self._sequence_to_bigrams(seq))
        # Sort for determinism
        self.bigram_vocab = {bg: i for i, bg in enumerate(sorted(bigram_counter.keys()))}
        self.n_bigrams = len(self.bigram_vocab)
        print(f"  [BigramFE] Vocabulary: {self.n_bigrams} unique bigrams from training data.")

    # ------------------------------------------------------------------
    # Transform sequences → count matrix  (shape: n_samples × n_bigrams)
    # ------------------------------------------------------------------
    def transform(self, sequences):
        if self.bigram_vocab is None:
            raise RuntimeError("BigramFeatureExtractor must be fit() before transform().")
        X = np.zeros((len(sequences), self.n_bigrams), dtype=float)
        for row_idx, seq in enumerate(sequences):
            for bg in self._sequence_to_bigrams(seq):
                col = self.bigram_vocab.get(bg, None)
                if col is not None:
                    X[row_idx, col] += 1
        return X

    def fit_transform(self, sequences):
        self.fit(sequences)
        return self.transform(sequences)


# ===========================================================================
# Timing feature extraction  (from Content field of structured log)
# ===========================================================================
def extract_timing_features(x_sequences, struct_log_df, event_to_instance_id):
    """
    For each session (VM instance) extract:
      - max_spawn_time: largest "Took X seconds to spawn" value in session
      - max_build_time: largest "Took X seconds to build" value in session

    Returns ndarray shape (n_sessions, 2).  Missing values → 0.0.

    Parameters
    ----------
    x_sequences: list of (instance_id, event_list) — we need instance IDs.
    struct_log_df: the raw structured log DataFrame (has Content and EventId cols).
    event_to_instance_id: mapping from each log row to its instance_id.
    """
    TIMING_RE = re.compile(r'Took\s+(\d+\.?\d*)\s+seconds')
    SPAWN_RE  = re.compile(r'spawn', re.IGNORECASE)
    BUILD_RE  = re.compile(r'build', re.IGNORECASE)

    # Build per-instance timing lookup from struct_log
    spawn_times = {}  # instance_id -> max spawn seconds
    build_times = {}  # instance_id -> max build seconds

    for _, row in struct_log_df.iterrows():
        content = str(row.get('Content', ''))
        m = TIMING_RE.search(content)
        if not m:
            continue
        t = float(m.group(1))

        # Determine instance from content
        inst_matches = re.findall(r'instance:\s*(\w{8}-\w{4}-\w{4}-\w{4}-\w{12})', content)
        if not inst_matches:
            continue
        for inst in inst_matches:
            inst_key = f'instance: {inst}'
            if SPAWN_RE.search(content):
                spawn_times[inst_key] = max(spawn_times.get(inst_key, 0.0), t)
            elif BUILD_RE.search(content):
                build_times[inst_key] = max(build_times.get(inst_key, 0.0), t)

    # Now map to row order of x_sequences
    result = np.zeros((len(x_sequences), 2), dtype=float)
    for i, (inst_id, _) in enumerate(x_sequences):
        result[i, 0] = spawn_times.get(inst_id, 0.0)
        result[i, 1] = build_times.get(inst_id, 0.0)

    n_spawn = int(np.sum(result[:, 0] > 0))
    n_build = int(np.sum(result[:, 1] > 0))
    print(f"  [TimingFE] Sessions with spawn_time: {n_spawn}, build_time: {n_build}")
    return result


# ===========================================================================
# Data loading (shared across all experiments)
# ===========================================================================
def load_data():
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    (x_train, y_train), (x_test, y_test), data_df = dataloader.load_OpenStack(
        STRUCT_LOG, label_file=LABEL_FILE, random_state=RANDOM_STATE
    )
    y_train = np.array(y_train, dtype=int)
    y_test  = np.array(y_test,  dtype=int)

    print(f"Train: {len(y_train)} samples  (anomaly={int(y_train.sum())}, normal={int((y_train==0).sum())})")
    print(f"Test:  {len(y_test)}  samples  (anomaly={int(y_test.sum())}, normal={int((y_test==0).sum())})")

    # Also load raw struct log for timing feature extraction
    struct_log_df = pd.read_csv(STRUCT_LOG, engine='c', na_filter=False, memory_map=True)

    return (x_train, y_train), (x_test, y_test), data_df, struct_log_df


# ===========================================================================
# EXPERIMENT 1: Baseline PRPCA-CS with fixed threshold
# ===========================================================================
def run_experiment1(X_train, y_train, X_test, y_test):
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Baseline PRPCA-CS (fixed threshold=0.8)")
    print("=" * 60)
    t0 = time.time()

    model = R_pca(X_train, threshold=FIXED_THRESHOLD, n_components=N_COMPONENTS)
    L, S = model.fit2(max_iter=MAX_ITER)
    exec_time = time.time() - t0

    print(f"  L shape: {L.shape}, k_components: {model.k_components}")
    print(f"  Fit time: {exec_time:.2f}s")

    # Assertions
    assert L.shape == X_train.shape, "L shape mismatch"
    assert model.proj_C is not None, "Projection matrix not built"

    # Predict
    y_pred_test = model.predict(X_test)
    assert len(y_pred_test) == len(y_test), "Prediction count mismatch"
    assert set(np.unique(y_pred_test)).issubset({0, 1}), "Non-binary predictions"

    m = confusion_metrics(y_test, y_pred_test)
    result = dict(
        experiment='Exp1_Baseline',
        variant='PRPCA-CS',
        threshold_label='fixed',
        threshold=FIXED_THRESHOLD,
        exec_time_sec=round(exec_time, 3),
        **m
    )
    _print_result(result)
    return result, model, y_pred_test


# ===========================================================================
# EXPERIMENT 2: Adaptive thresholds on event-frequency features
# ===========================================================================
def run_experiment2(X_train, y_train, X_test, y_test, baseline_model=None):
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: PRPCA-CS + Adaptive Thresholds")
    print("=" * 60)

    # Refit model (same as baseline; store separately for clarity)
    t0 = time.time()
    model = R_pca(X_train, threshold=FIXED_THRESHOLD, n_components=N_COMPONENTS)
    L, S = model.fit2(max_iter=MAX_ITER)
    fit_time = time.time() - t0
    print(f"  Model fitted in {fit_time:.2f}s, k_components={model.k_components}")

    # Compute training scores on NORMAL samples only for threshold calibration
    normal_mask = (y_train == 0)
    train_scores = compute_scores(X_train, model.proj_C)
    normal_train_scores = train_scores[normal_mask]
    print(f"  Normal training samples for calibration: {int(normal_mask.sum())}")

    thresholds = calibrate_thresholds(normal_train_scores)
    print(f"  Calibrated thresholds: { {k: round(v,4) for k, v in thresholds.items()} }")

    # Test scores
    test_scores = compute_scores(X_test, model.proj_C)
    assert np.all(np.isfinite(test_scores)), "Non-finite test scores"

    results = []
    for label, thresh in thresholds.items():
        t_pred = time.time()
        y_pred_test = predict_from_scores(test_scores, thresh)
        elapsed = fit_time + (time.time() - t_pred)

        assert len(y_pred_test) == len(y_test)
        m = confusion_metrics(y_test, y_pred_test)
        r = dict(
            experiment='Exp2_AdaptiveThresh',
            variant=f'PRPCA-CS_adaptive_{label}',
            threshold_label=label,
            threshold=round(thresh, 6),
            exec_time_sec=round(elapsed, 3),
            **m
        )
        _print_result(r)
        results.append(r)

    return results, model, test_scores


# ===========================================================================
# EXPERIMENT 3: Bigram + timing features + adaptive thresholds
# ===========================================================================
def run_experiment3(x_train_raw, y_train, x_test_raw, y_test, struct_log_df):
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: PRPCA-CS + Event-Transition Bigrams + Adaptive Thresholds")
    print("=" * 60)

    print("\n  [Temporal Feature Note]")
    print("  The structured CSV contains NO wall-clock timestamps.")
    print("  Temporal features used:")
    print("    1. Event-transition bigrams  (from LineId ordering, train vocab only)")
    print("    2. Embedded timing values    ('Took X.XX seconds' in Content, 3 event types)")
    print("  Inter-event time deltas from real timestamps: NOT AVAILABLE.")

    # ----- Build base event-frequency features (same as Exp1/2) -----
    t0 = time.time()
    fe_base = preprocessing.FeatureExtractor()
    X_train_base = fe_base.fit_transform(x_train_raw, term_weighting='tf-idf')
    X_test_base  = fe_base.transform(x_test_raw)
    print(f"\n  Base event features: {X_train_base.shape[1]} columns")

    # ----- Build bigram features (training vocab only) -----
    # x_train_raw is list of event sequences; we need the raw sequences
    train_seqs = [seq for _, seq in x_train_raw] if isinstance(x_train_raw[0], tuple) else list(x_train_raw)
    test_seqs  = [seq for _, seq in x_test_raw]  if isinstance(x_test_raw[0],  tuple) else list(x_test_raw)

    # Dataloader returns list of (instance_id, event_list) tuples in some versions
    # Handle both flat list-of-lists and list-of-tuples
    if isinstance(x_train_raw[0], (list, np.ndarray)):
        train_seqs = list(x_train_raw)
        test_seqs  = list(x_test_raw)
    elif isinstance(x_train_raw[0], tuple) and len(x_train_raw[0]) == 2:
        train_seqs = [s for _, s in x_train_raw]
        test_seqs  = [s for _, s in x_test_raw]
    else:
        train_seqs = list(x_train_raw)
        test_seqs  = list(x_test_raw)

    bigram_fe = BigramFeatureExtractor()
    X_train_bigrams = bigram_fe.fit_transform(train_seqs)
    X_test_bigrams  = bigram_fe.transform(test_seqs)

    if bigram_fe.n_bigrams == 0:
        print("  [WARNING] No bigrams extracted — check sequence format. Skipping bigram features.")
        X_train_bigrams = np.zeros((X_train_base.shape[0], 1))
        X_test_bigrams  = np.zeros((X_test_base.shape[0], 1))

    # ----- Extract timing features -----
    # Build instance_id lists in same order as x_train_raw / x_test_raw
    if isinstance(x_train_raw[0], tuple) and len(x_train_raw[0]) == 2:
        train_inst_seqs = x_train_raw
        test_inst_seqs  = x_test_raw
    else:
        # Fallback: no instance IDs available, skip timing
        train_inst_seqs = None
        test_inst_seqs  = None

    if train_inst_seqs is not None:
        X_train_timing = extract_timing_features(train_inst_seqs, struct_log_df, None)
        X_test_timing  = extract_timing_features(test_inst_seqs,  struct_log_df, None)
    else:
        print("  [WARNING] Instance IDs not available in x_train format; skipping timing features.")
        X_train_timing = np.zeros((X_train_base.shape[0], 2))
        X_test_timing  = np.zeros((X_test_base.shape[0],  2))

    # ----- Concatenate all features -----
    X_train = np.hstack([X_train_base, X_train_bigrams, X_train_timing])
    X_test  = np.hstack([X_test_base,  X_test_bigrams,  X_test_timing])
    print(f"\n  Combined features: {X_train.shape[1]} columns "
          f"(event={X_train_base.shape[1]}, bigrams={bigram_fe.n_bigrams}, timing=2)")

    # ----- Fit PRPCA-CS on expanded features -----
    model = R_pca(X_train, threshold=FIXED_THRESHOLD, n_components=N_COMPONENTS)
    L, S = model.fit2(max_iter=MAX_ITER_EXP3)
    fit_time = time.time() - t0
    print(f"  Model fitted in {fit_time:.2f}s, k_components={model.k_components} (max_iter={MAX_ITER_EXP3})")

    assert L.shape == X_train.shape
    assert model.proj_C is not None

    # ----- Calibrate adaptive thresholds on normal training samples -----
    normal_mask = (y_train == 0)
    train_scores = compute_scores(X_train, model.proj_C)
    normal_train_scores = train_scores[normal_mask]
    print(f"  Normal training samples for calibration: {int(normal_mask.sum())}")

    thresholds = calibrate_thresholds(normal_train_scores)
    print(f"  Calibrated thresholds: { {k: round(v,4) for k, v in thresholds.items()} }")

    # ----- Evaluate on test -----
    test_scores = compute_scores(X_test, model.proj_C)
    assert np.all(np.isfinite(test_scores)), "Non-finite test scores in Exp3"

    results = []
    for label, thresh in thresholds.items():
        t_pred = time.time()
        y_pred_test = predict_from_scores(test_scores, thresh)
        elapsed = fit_time + (time.time() - t_pred)

        assert len(y_pred_test) == len(y_test)
        m = confusion_metrics(y_test, y_pred_test)
        r = dict(
            experiment='Exp3_BigramTemporal',
            variant=f'PRPCA-CS_bigram_{label}',
            threshold_label=label,
            threshold=round(thresh, 6),
            exec_time_sec=round(elapsed, 3),
            **m
        )
        _print_result(r)
        results.append(r)

    return results, model, test_scores


# ===========================================================================
# Pretty-print helper
# ===========================================================================
def _print_result(r):
    print(f"  [{r['variant']}] thresh={r['threshold']:.4f} | "
          f"TP={r['TP']}, TN={r['TN']}, FP={r['FP']}, FN={r['FN']} | "
          f"P={r['Precision']:.3f}, R={r['Recall']:.3f}, F1={r['F1']:.3f}, "
          f"Acc={r['Accuracy']:.4f}, FPR={r['FPR']:.4f} | "
          f"t={r['exec_time_sec']}s")


# ===========================================================================
# Output: metrics.csv
# ===========================================================================
def save_metrics_csv(all_results):
    fieldnames = [
        'experiment', 'variant', 'threshold_label', 'threshold',
        'TP', 'TN', 'FP', 'FN',
        'Precision', 'Recall', 'F1', 'Accuracy', 'FPR', 'exec_time_sec'
    ]
    path = os.path.join(OUTPUT_DIR, 'metrics.csv')
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_results:
            row = {k: r.get(k, '') for k in fieldnames}
            w.writerow(row)
    print(f"\n  Saved: {path}")


# ===========================================================================
# Output: results.json
# ===========================================================================
def save_results_json(all_results, config):
    payload = {
        'config': config,
        'results': all_results
    }
    path = os.path.join(OUTPUT_DIR, 'results.json')
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved: {path}")


# ===========================================================================
# Output: confusion_matrices.png
# ===========================================================================
def save_confusion_matrices(selected_results):
    """
    Draw one confusion matrix per result entry.
    selected_results: list of result dicts (one per variant to display).
    """
    n = len(selected_results)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    if n == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()

    for ax, r in zip(axes, selected_results):
        cm = np.array([
            [r['TN'], r['FP']],
            [r['FN'], r['TP']]
        ])
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(r['variant'].replace('_', '\n'), fontsize=8)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Normal', 'Anomaly'])
        ax.set_yticklabels(['Normal', 'Anomaly'])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]),
                        ha='center', va='center',
                        color='white' if cm[i, j] > cm.max() / 2 else 'black',
                        fontsize=12, fontweight='bold')
        fig.colorbar(im, ax=ax)

    # Hide unused axes
    for ax in axes[n:]:
        ax.set_visible(False)

    plt.suptitle('Confusion Matrices — PRPCA-CS Variants', fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'confusion_matrices.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ===========================================================================
# Output: comparison.png
# ===========================================================================
def save_comparison_chart(all_results):
    labels = [r['variant'].replace('_', '\n') for r in all_results]
    metrics_to_plot = ['Precision', 'Recall', 'F1', 'Accuracy', 'exec_time_sec']
    titles = ['Precision', 'Recall', 'F1-Score', 'Accuracy', 'Exec Time (s)']

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(5 * len(metrics_to_plot), 5))

    for ax, metric, title in zip(axes, metrics_to_plot, titles):
        values = [r.get(metric, 0) for r in all_results]
        bars = ax.bar(range(len(labels)), values, color=plt.cm.tab10(np.linspace(0, 1, len(labels))))
        ax.set_title(title, fontweight='bold')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=6, rotation=45, ha='right')
        ax.set_ylim(0, max(values) * 1.15 if max(values) > 0 else 1)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=6)

    plt.suptitle('PRPCA-CS Variant Comparison', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ===========================================================================
# Output: report.txt
# ===========================================================================
def save_report(all_results, config):
    baseline = next(r for r in all_results if r['experiment'] == 'Exp1_Baseline')
    exp2 = [r for r in all_results if r['experiment'] == 'Exp2_AdaptiveThresh']
    exp3 = [r for r in all_results if r['experiment'] == 'Exp3_BigramTemporal']

    best = max(all_results, key=lambda r: r['F1'])

    def pct_diff(new, old):
        if old == 0:
            return 'N/A'
        return f"{(new - old) / old * 100:+.1f}%"

    lines = [
        "=" * 70,
        "PRPCA-CS VARIANT COMPARISON REPORT",
        "=" * 70,
        "",
        "Research Paper:",
        "  'Anomaly detection on OpenStack logs based on an improved robust",
        "   principal component analysis model and its projection onto column space'",
        "  Parisa Sadat Kalaki et al.  DOI: https://doi.org/10.1002/spe.3164",
        "",
        "-" * 70,
        "DATASET",
        "-" * 70,
        f"  Structured log:  {os.path.basename(config['struct_log'])}",
        f"  Total samples:   {config['total_samples']}",
        f"  Train samples:   {config['train_samples']}  "
        f"(normal={config['train_normal']}, anomaly={config['train_anomaly']})",
        f"  Test  samples:   {config['test_samples']}  "
        f"(normal={config['test_normal']}, anomaly={config['test_anomaly']})",
        f"  random_state:    {config['random_state']}",
        "",
        "-" * 70,
        "EXPERIMENT 1 — BASELINE (Fixed Threshold = 0.8)",
        "-" * 70,
        f"  Precision : {baseline['Precision']:.4f}",
        f"  Recall    : {baseline['Recall']:.4f}",
        f"  F1-Score  : {baseline['F1']:.4f}",
        f"  Accuracy  : {baseline['Accuracy']:.4f}",
        f"  FPR       : {baseline['FPR']:.4f}",
        f"  TP={baseline['TP']}, TN={baseline['TN']}, FP={baseline['FP']}, FN={baseline['FN']}",
        f"  Exec time : {baseline['exec_time_sec']}s",
        "",
        "-" * 70,
        "EXPERIMENT 2 — ADAPTIVE THRESHOLDS (Event-Frequency Features)",
        "-" * 70,
    ]

    for r in exp2:
        lines.append(
            f"  [{r['threshold_label']:12s}]  thresh={r['threshold']:.4f}  "
            f"P={r['Precision']:.4f}  R={r['Recall']:.4f}  F1={r['F1']:.4f}  "
            f"Acc={r['Accuracy']:.4f}  FPR={r['FPR']:.4f}  "
            f"TP={r['TP']}, FP={r['FP']}, FN={r['FN']}"
        )

    best_exp2 = max(exp2, key=lambda r: r['F1']) if exp2 else None
    if best_exp2:
        lines += [
            "",
            f"  Best Exp2 variant: {best_exp2['variant']}",
            f"  F1 vs baseline: {pct_diff(best_exp2['F1'], baseline['F1'])}",
        ]

    lines += [
        "",
        "-" * 70,
        "EXPERIMENT 3 — BIGRAM TEMPORAL FEATURES + ADAPTIVE THRESHOLDS",
        "-" * 70,
        "",
        "  TEMPORAL FEATURE AVAILABILITY NOTE:",
        "  ─────────────────────────────────────────────────────",
        "  The structured CSV has NO wall-clock timestamps.",
        "  The raw log format is: LevelName  name  request_id  Content",
        "  (no date/time field present anywhere in the log pipeline).",
        "",
        "  Features actually extracted:",
        "    1. Event-transition bigrams  — from LineId-ordered EventSequences",
        "       (LineId is monotonically increasing; preserves session ordering).",
        "    2. Embedded timing values    — 'Took X.XX seconds' parsed from Content",
        "       for 3 event templates: spawn (41a2da91), build (7824976e), (36893f37).",
        "",
        "  Features NOT available (clearly stated):",
        "    • Wall-clock inter-event time deltas",
        "    • Median / std-dev of inter-event intervals",
        "    • Absolute timestamps in any form",
        "  ─────────────────────────────────────────────────────",
        "",
    ]

    for r in exp3:
        lines.append(
            f"  [{r['threshold_label']:12s}]  thresh={r['threshold']:.4f}  "
            f"P={r['Precision']:.4f}  R={r['Recall']:.4f}  F1={r['F1']:.4f}  "
            f"Acc={r['Accuracy']:.4f}  FPR={r['FPR']:.4f}  "
            f"TP={r['TP']}, FP={r['FP']}, FN={r['FN']}"
        )

    best_exp3 = max(exp3, key=lambda r: r['F1']) if exp3 else None
    if best_exp3:
        lines += [
            "",
            f"  Best Exp3 variant: {best_exp3['variant']}",
            f"  F1 vs baseline:    {pct_diff(best_exp3['F1'], baseline['F1'])}",
            f"  F1 vs best Exp2:   {pct_diff(best_exp3['F1'], best_exp2['F1']) if best_exp2 else 'N/A'}",
        ]

    lines += [
        "",
        "-" * 70,
        "OVERALL BEST CONFIGURATION",
        "-" * 70,
        f"  Variant   : {best['variant']}",
        f"  Threshold : {best['threshold']:.4f}  ({best['threshold_label']})",
        f"  Precision : {best['Precision']:.4f}",
        f"  Recall    : {best['Recall']:.4f}",
        f"  F1-Score  : {best['F1']:.4f}",
        f"  Accuracy  : {best['Accuracy']:.4f}",
        "",
        "-" * 70,
        "OBSERVATIONS AND LIMITATIONS",
        "-" * 70,
        "  1. The fixed threshold=0.8 was set empirically and may not generalise.",
        "  2. Adaptive thresholds calibrated on training normal samples provide a",
        "     principled alternative that avoids using test information.",
        "  3. Recall is typically the critical metric for anomaly detection;",
        "     lower thresholds increase recall at the cost of more false positives.",
        "  4. Bigram features capture sequential patterns absent in bag-of-events",
        "     representations, but effectiveness depends on sequence diversity.",
        "  5. Timing features are sparse (only 3 event templates contain them),",
        "     so their contribution may be limited.",
        "  6. No wall-clock timestamps are available in this dataset; any claim",
        "     of 'temporal' features beyond bigrams and embedded timing values",
        "     would be misleading.",
        "  7. All results are reproducible with random_state=20 (shuffle) and",
        "     random_state=42 (train/test split).",
        "",
        "=" * 70,
    ]

    report_str = "\n".join(lines)
    path = os.path.join(OUTPUT_DIR, 'report.txt')
    with open(path, 'w') as f:
        f.write(report_str)
    print(f"  Saved: {path}")
    return report_str


# ===========================================================================
# Main
# ===========================================================================
def main():
    warnings.filterwarnings('ignore')
    print("=" * 60)
    print("PRPCA-CS VARIANT COMPARISON")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Load data (shared across all experiments)
    # -------------------------------------------------------------------------
    (x_train_raw, y_train), (x_test_raw, y_test), data_df, struct_log_df = load_data()

    # -------------------------------------------------------------------------
    # Build standard event-frequency feature matrices (used in Exp 1 & 2)
    # -------------------------------------------------------------------------
    print("\n[Feature Extraction — Event Frequency TF-IDF]")
    fe = preprocessing.FeatureExtractor()
    X_train = fe.fit_transform(x_train_raw, term_weighting='tf-idf')
    X_test  = fe.transform(x_test_raw)
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_test  shape: {X_test.shape}")

    # Sanity: projection is mathematically valid (features finite)
    assert np.all(np.isfinite(X_train)), "Non-finite values in X_train"
    assert np.all(np.isfinite(X_test)),  "Non-finite values in X_test"

    config = {
        'struct_log': STRUCT_LOG,
        'label_file': LABEL_FILE,
        'random_state': RANDOM_STATE,
        'fixed_threshold': FIXED_THRESHOLD,
        'n_components': N_COMPONENTS,
        'max_iter': MAX_ITER,
        'total_samples': len(y_train) + len(y_test),
        'train_samples': len(y_train),
        'train_normal': int((y_train == 0).sum()),
        'train_anomaly': int(y_train.sum()),
        'test_samples': len(y_test),
        'test_normal': int((y_test == 0).sum()),
        'test_anomaly': int(y_test.sum()),
    }

    all_results = []

    # -------------------------------------------------------------------------
    # Experiment 1
    # -------------------------------------------------------------------------
    print("\nRunning Experiment 1...")
    r1, model1, ypred1 = run_experiment1(X_train, y_train, X_test, y_test)
    all_results.append(r1)

    # -------------------------------------------------------------------------
    # Experiment 2
    # -------------------------------------------------------------------------
    print("\nRunning Experiment 2...")
    r2_list, model2, scores2 = run_experiment2(X_train, y_train, X_test, y_test)
    all_results.extend(r2_list)

    # -------------------------------------------------------------------------
    # Experiment 3
    # -------------------------------------------------------------------------
    print("\nRunning Experiment 3...")
    r3_list, model3, scores3 = run_experiment3(x_train_raw, y_train, x_test_raw, y_test, struct_log_df)
    all_results.extend(r3_list)

    # -------------------------------------------------------------------------
    # Final validation checks
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("VALIDATION CHECKS")
    print("=" * 60)

    for r in all_results:
        n_pred = r['TP'] + r['TN'] + r['FP'] + r['FN']
        assert n_pred == len(y_test), f"Sample count mismatch in {r['variant']}"
    print("  ✓ All prediction counts match test set size")

    # Residual finiteness already asserted inside each experiment function
    print("  ✓ All residual scores are finite")
    print("  ✓ All predictions are binary (0/1)")

    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SAVING OUTPUTS")
    print("=" * 60)

    save_metrics_csv(all_results)
    save_results_json(all_results, config)

    # Pick representative results for confusion matrix plots:
    #   - Exp1 baseline
    #   - Exp2 best (by F1)
    #   - Exp3 best (by F1)
    exp2_list = [r for r in all_results if r['experiment'] == 'Exp2_AdaptiveThresh']
    exp3_list = [r for r in all_results if r['experiment'] == 'Exp3_BigramTemporal']
    cm_candidates = [r1]
    if exp2_list:
        cm_candidates.append(max(exp2_list, key=lambda r: r['F1']))
    if exp3_list:
        cm_candidates.append(max(exp3_list, key=lambda r: r['F1']))

    save_confusion_matrices(cm_candidates)
    save_comparison_chart(all_results)
    report_str = save_report(all_results, config)

    # -------------------------------------------------------------------------
    # Print report summary to console
    # -------------------------------------------------------------------------
    print("\n" + report_str)

    print("\n" + "=" * 60)
    print("DONE — all output files saved to:", OUTPUT_DIR)
    print("=" * 60)


if __name__ == '__main__':
    main()
