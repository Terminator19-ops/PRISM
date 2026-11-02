# Docker Deployment & REST API Guide: PRPCA-CS Anomaly Detection

This document provides instructions for building, running, operating, and querying the Dockerised **PRPCA-CS (Projection-based Robust Principal Component Analysis onto Column Space)** log anomaly detection service.

---

## 1. Application Overview

The PRPCA-CS Anomaly Detection Service packages the research implementation of OpenStack Nova log anomaly detection into a production-ready **FastAPI** REST API.

The service performs:
1. **Input Validation:** Ingests structured OpenStack CSV log files, verifying that required session identifiers and parsed event templates are present.
2. **Session Aggregation:** Extracts individual VM instance execution traces using regex matching (`instance: <uuid>`).
3. **TF-IDF Feature Mapping:** Transforms variable-length event traces into fixed-dimensional event-frequency vectors ($28$ features) using the pre-fit vocabulary and IDF weights.
4. **Subspace Projection:** Projects incoming session vectors onto the normal execution subspace via precomputed projection matrix $P_C = V_k V_k^T$ ($k=4$).
5. **Residual Anomaly Decision:** Computes sparse residual deviation vectors $S_{\text{proj}} = X - X P_C$ and flags anomalous sessions when any event residual exceeds the threshold ($\tau = 0.8$ baseline default, or custom user override).

All heavy matrix decomposition is performed ahead-of-time during artifact preparation. The Docker service operates in pure inference mode with **sub-millisecond per-session latency**.

---

## 2. Project Architecture

```
OpenStack/
├── app/
│   ├── __init__.py
│   ├── main.py                     # FastAPI application routes (/health, /predict)
│   ├── inference.py                # Preprocessing validation & PRPCA-CS inference engine
│   └── schemas.py                  # Pydantic request and response schemas
│
├── model/
│   └── prpca_model.joblib          # Exported model artifacts (proj_C, k_components, TF-IDF weights)
│
├── loglizer/                       # Core research library
│   ├── loglizer/models/RPCA.py     # PRPCA-CS mathematical model
│   ├── loglizer/preprocessing.py   # Feature extraction and TF-IDF
│   └── loglizer/dataloader.py      # Session windowing and parsing logic
│
├── tests/
│   └── data/
│       ├── sample_valid_sessions.csv    # Valid OpenStack structured CSV test sample
│       ├── sample_missing_columns.csv   # Invalid CSV test case
│       └── sample_raw_unparsed.log      # Raw log test case
│
├── train_and_save_model.py         # Offline training & artifact serialization script
├── validate_docker_predictions.py  # End-to-end parity validation test suite
├── Dockerfile                      # Production container definition
├── .dockerignore                   # Build context filter
├── requirements.txt                # Production Python dependencies
└── README-Docker.md                # This documentation
```

### Packaging & Isolation Strategy
- **Base Image:** `python:3.12-slim` to minimize image size and attack surface.
- **Artifact Decoupling:** Training data (`25,000+` lines) is omitted from the image via `.dockerignore`. Only the lightweight, trained model artifact (`prpca_model.joblib`, ~8.4 KB) is bundled.
- **Security:** Runs as an unprivileged user (`appuser`, UID 1000). Root access is revoked after package installation.
- **Health Checks:** Uses native Docker container health checks hitting `/health` via `curl`.

---

## 3. Prerequisites

- **Docker:** Version 20.10+ (Docker Desktop, Docker Engine, or Podman)
- **Host Ports:** Port `8000` available on the host machine
- **Disk Space:** ~1 GB for the Docker image

---

## 4. Docker Image Build Instructions

Build the image tagged `prpca-detector:v1` from the project root:

```bash
docker build -t prpca-detector:v1 .
```

To verify the image was created:

```bash
docker images prpca-detector
```

Expected output:
```text
IMAGE               ID             DISK USAGE   CONTENT SIZE
prpca-detector:v1   ddb27f3da1ec        858MB          198MB
```

---

## 5. Container Run Instructions

### Run in Background (Detached Mode)

```bash
docker run -d --name prpca-service -p 8000:8000 prpca-detector:v1
```

### Run Interactively with Terminal Output

```bash
docker run --rm -it -p 8000:8000 prpca-detector:v1
```

### Check Container Status & Logs

```bash
docker ps --filter "name=prpca-service"
docker logs -f prpca-service
```

---

## 6. API Endpoint Documentation

### A. Health Check: `GET /health`

Verifies that the FastAPI application and the PRPCA-CS inference engine are loaded and ready to serve requests.

- **URL:** `http://localhost:8000/health`
- **Method:** `GET`
- **Response Format:** `application/json`

#### Response:
```json
{
  "status": "healthy",
  "service": "PRPCA-CS Anomaly Detection"
}
```

---

### B. Anomaly Prediction: `POST /predict`

Accepts a structured log CSV file via multipart file upload, extracts VM sessions, applies the learned normal subspace projection, and returns anomaly classifications.

- **URL:** `http://localhost:8000/predict`
- **Method:** `POST`
- **Content-Type:** `multipart/form-data`
- **Parameters:**
  - `file` (form-data file, required): The structured OpenStack CSV log.
  - `threshold` (query float, optional): Custom anomaly cutoff (defaults to baseline `0.8`).

#### Supported Input Format
The uploaded file must be a structured CSV containing at minimum:
1. `Content`: Log text containing session identifiers matching `instance: <uuid>` (e.g. `[instance: 11760334-ac63-4cc8-9086-578422af8c99]`).
2. `EventId`: Parsed event template identifier (e.g. `8d02d411`).

#### Success Response (`200 OK`):
```json
{
  "total_sessions": 20,
  "normal_sessions": 20,
  "anomalous_sessions": 0,
  "anomaly_rate": 0.0,
  "threshold_used": 0.8,
  "predictions": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  "session_details": [
    {
      "instance_id": "instance: 11760334-ac63-4cc8-9086-578422af8c99",
      "prediction": 0,
      "status": "Normal",
      "max_residual": 0.101824
    }
  ],
  "model_info": {
    "model_name": "PRPCA-CS",
    "k_components": 4,
    "num_features": 28,
    "default_threshold": 0.8
  }
}
```

#### Error Response (`400 Bad Request`):
When an invalid or unparsed file is uploaded:
```json
{
  "detail": "Missing required columns in CSV: ['EventId']. Available columns: ['LineId', 'LevelName', 'Content']. Supported format: Structured CSV log file containing at least two columns: 'Content' (with session identifiers matching 'instance: <uuid>') and 'EventId' (parsed template ID, e.g., '8d02d411'). Raw unstructured log files must first be processed through a log parser such as IPLoM or Drain before submitting for anomaly detection."
}
```

---

## 7. Example `curl` Requests

### 1. Check Service Health
```bash
curl -i http://localhost:8000/health
```

### 2. Predict on Valid Sample File (Baseline Threshold = 0.8)
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@tests/data/sample_valid_sessions.csv"
```

### 3. Predict with Custom Adaptive Threshold (e.g., 0.2037)
```bash
curl -X POST "http://localhost:8000/predict?threshold=0.2037" \
  -F "file=@tests/data/sample_valid_sessions.csv"
```

### 4. Test Error Handling on Raw Unparsed Log
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@tests/data/sample_raw_unparsed.log"
```

---

## 8. Docker Operations Guide

The following standard lifecycle operations were verified:

### 1. List Images
```bash
docker images prpca-detector
```

### 2. List Running Containers
```bash
docker ps --filter "name=prpca-service"
```

### 3. Inspect Container & Image Configuration
```bash
docker inspect prpca-detector:v1
```

### 4. Tag the Image (Version 2)
```bash
docker tag prpca-detector:v1 prpca-detector:v2
```

### 5. Export / Save the Image Archive
```bash
docker save -o prpca-detector.tar prpca-detector:v1
```

### 6. Remove Tagged Image
```bash
docker rmi prpca-detector:v1
```

### 7. Restore Image from Archive
```bash
docker load -i prpca-detector.tar
```

---

## 9. Parity Validation Between Python and Docker

To prove mathematical and operational equivalence:
- Direct Python inference was run using `PRPCAInferenceEngine`.
- The exact same input payload was dispatched to `http://localhost:8000/predict`.

Run the automated validation script:
```bash
./venv/bin/python validate_docker_predictions.py
```

### Verification Results:
| Metric | Local Python | Docker Container API | Status |
|:---|:---:|:---:|:---:|
| Total Sessions | 20 | 20 | **100% Match** |
| Normal Sessions ($\tau=0.8$) | 20 | 20 | **100% Match** |
| Anomalous Sessions ($\tau=0.8$) | 0 | 0 | **100% Match** |
| Normal Sessions ($\tau=0.2$) | 19 | 19 | **100% Match** |
| Anomalous Sessions ($\tau=0.2$) | 1 | 1 | **100% Match** |
| Max Residual Precision | Floating-point exact | Floating-point exact | **100% Match** |

---

## 10. Troubleshooting & Known Limitations

### 1. Raw Logs vs Structured Logs
- **Limitation:** The API accepts *structured CSV logs*, not raw `.log` text files.
- **Why:** OpenStack logs must be parsed into event templates (e.g., via IPLoM or Drain) before session vectors can be constructed. Submitting raw unstructured logs will result in an informative `400 Bad Request` explaining this requirement.

### 2. Port Binding Conflicts
- If port `8000` is already in use by another application:
  ```bash
  docker run -d --name prpca-service -p 8080:8000 prpca-detector:v1
  ```
  Then query `http://localhost:8080/health`.

### 3. Updating the Model Artifact
- If hyperparameters or the training dataset are updated, regenerate the model artifact:
  ```bash
  python train_and_save_model.py
  docker build -t prpca-detector:v1 .
  ```

