# -*- coding: utf-8 -*-
"""
app/main.py
===========
FastAPI application exposing PRPCA-CS OpenStack log anomaly detection
through REST endpoints.
"""

import os
import sys
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Query, HTTPException, status
from fastapi.responses import JSONResponse

# Add project root and loglizer to sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
LOGLIZER_DIR = os.path.join(PROJECT_ROOT, 'loglizer')
if LOGLIZER_DIR not in sys.path:
    sys.path.insert(0, LOGLIZER_DIR)

from app.schemas import HealthResponse, PredictionResponse, ErrorResponse
from app.inference import PRPCAInferenceEngine, InvalidLogFileError

# Global engine instance
inference_engine: Optional[PRPCAInferenceEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle event handler for model loading on application startup."""
    global inference_engine
    model_path = os.getenv(
        "MODEL_PATH",
        os.path.join(PROJECT_ROOT, "model", "prpca_model.joblib")
    )
    print(f"Starting PRPCA-CS API... Initializing model from {model_path}")
    try:
        inference_engine = PRPCAInferenceEngine(model_path=model_path)
    except Exception as e:
        print(f"ERROR loading model: {e}")
        raise e
    yield
    print("Shutting down PRPCA-CS API service.")


app = FastAPI(
    title="PRPCA-CS OpenStack Log Anomaly Detection API",
    description=(
        "Production REST service for detecting anomalies in OpenStack Nova logs "
        "using Robust Principal Component Analysis with Column Space Projection (PRPCA-CS)."
    ),
    version="1.0.0",
    lifespan=lifespan
)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check endpoint",
    tags=["Monitoring"]
)
async def health_check():
    """Returns the operational status of the service."""
    return {
        "status": "healthy",
        "service": "PRPCA-CS Anomaly Detection"
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file format or missing columns"},
        500: {"model": ErrorResponse, "description": "Internal server/inference error"}
    },
    summary="Predict anomalies from an uploaded structured log file",
    tags=["Inference"]
)
async def predict_anomalies(
    file: UploadFile = File(..., description="OpenStack structured log CSV file"),
    threshold: Optional[float] = Query(
        None,
        description="Optional custom anomaly threshold (defaults to baseline 0.8)",
        ge=0.0
    )
):
    """
    Accepts a structured OpenStack log CSV file, extracts VM session sequences,
    transforms them via the learned TF-IDF feature vocabulary, projects onto the
    learned normal subspace, and classifies anomalous sessions based on residual error.
    """
    if inference_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not initialized yet."
        )

    # Basic filename validation
    filename = file.filename or "unknown"
    if not (filename.endswith(".csv") or filename.endswith(".log") or filename.endswith(".txt")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file extension '{filename}'. Please upload a structured CSV log file "
                "with 'Content' and 'EventId' columns."
            )
        )

    try:
        file_bytes = await file.read()
        result = inference_engine.predict_from_file(
            file_content=file_bytes,
            threshold_override=threshold
        )
        return result

    except InvalidLogFileError as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(err)
        )
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline execution error: {str(err)}"
        )
