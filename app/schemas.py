# -*- coding: utf-8 -*-
"""
app/schemas.py
==============
Pydantic schemas for the FastAPI service requests and responses.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., example="healthy")
    service: str = Field(..., example="PRPCA-CS Anomaly Detection")


class SessionDetail(BaseModel):
    instance_id: str = Field(..., description="Unique OpenStack VM instance identifier")
    prediction: int = Field(..., description="0 for normal, 1 for anomaly")
    status: str = Field(..., description="'Normal' or 'Anomaly'")
    max_residual: float = Field(..., description="Maximum residual error across event features")


class PredictionResponse(BaseModel):
    total_sessions: int = Field(..., description="Total VM sessions evaluated")
    normal_sessions: int = Field(..., description="Total sessions classified as normal")
    anomalous_sessions: int = Field(..., description="Total sessions classified as anomalous")
    anomaly_rate: float = Field(..., description="Proportion of anomalous sessions")
    threshold_used: float = Field(..., description="Anomaly classification threshold applied")
    predictions: List[int] = Field(..., description="Binary prediction array (0=Normal, 1=Anomaly)")
    session_details: Optional[List[SessionDetail]] = Field(None, description="Detailed per-session predictions")
    model_info: Optional[Dict[str, Any]] = Field(None, description="Model metadata and subspace parameters")


class ErrorResponse(BaseModel):
    detail: str
    error_type: str
    supported_format: str
