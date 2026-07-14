"""
FastAPI endpoints for Quantum Dengue STPP predictions.

Provides REST API for:
- Health check
- Model predictions
- Batch forecasting
- Metrics & monitoring
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from typing import List, Optional, Dict, Any
from datetime import datetime
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app with metadata
app = FastAPI(
    title="Quantum Dengue STPP API",
    description="Quantum-Enhanced Spatio-Temporal Point Process for Dengue Prediction",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# =============================================================================
# Request/Response Models
# =============================================================================

class LocationInput(BaseModel):
    """Geographic location input."""
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)

    @field_validator('latitude', 'longitude')
    @classmethod
    def validate_coords(cls, v):
        if v is None or np.isnan(v):
            raise ValueError("Invalid coordinates")
        return v


class PredictionRequest(BaseModel):
    """Request for single prediction."""
    location: LocationInput
    forecast_horizon: int = Field(default=3, ge=1, le=12)
    model_version: Optional[str] = "latest"


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions."""
    locations: List[LocationInput]
    forecast_horizon: int = Field(default=3, ge=1, le=12)
    model_version: Optional[str] = "latest"


class PredictionResponse(BaseModel):
    """Response for prediction."""
    location: Dict[str, float]
    predictions: Dict[str, float]
    confidence: Dict[str, float]
    risk_level: str
    model_version: str
    timestamp: str


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""
    predictions: List[PredictionResponse]
    total_count: int
    high_risk_count: int
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    gpu_available: bool
    version: str


# =============================================================================
# Mock Model (replace with actual model loading)
# =============================================================================

class DenguePredictor:
    """
    Dengue prediction model wrapper.
    Replace this with actual model loading.
    """
    
    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.version = "1.0.0"
        self.is_loaded = False
    
    def load_model(self, path: Optional[str] = None):
        """Load the trained model."""
        # TODO: Load actual model
        # self.model = load_trained_model(path)
        self.is_loaded = True
        logger.info(f"Model loaded on {self.device}")
    
    def predict(
        self,
        lat: float,
        lon: float,
        horizon: int = 3
    ) -> Dict[str, Any]:
        """
        Generate prediction for a location.
        
        Returns dict with predictions, confidence, and risk level.
        """
        # Mock prediction (replace with actual model inference)
        base_rate = np.random.uniform(0.5, 2.0)
        
        predictions = {}
        confidence = {}
        
        for month in range(1, horizon + 1):
            # Simulate monthly predictions
            pred = base_rate * (1 + 0.1 * np.sin(month / 12 * 2 * np.pi))
            predictions[f"month_{month}"] = round(float(pred), 2)
            confidence[f"month_{month}"] = round(float(np.random.uniform(0.7, 0.95)), 2)
        
        # Calculate risk level
        avg_pred = np.mean(list(predictions.values()))
        if avg_pred > 1.5:
            risk_level = "HIGH"
        elif avg_pred > 0.8:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return {
            "predictions": predictions,
            "confidence": confidence,
            "risk_level": risk_level
        }
    
    def batch_predict(
        self,
        locations: List[tuple],
        horizon: int = 3
    ) -> List[Dict[str, Any]]:
        """Generate predictions for multiple locations."""
        return [
            self.predict(lat, lon, horizon)
            for lat, lon in locations
        ]


# Global model instance
predictor = DenguePredictor()


# =============================================================================
# API Endpoints
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    predictor.load_model()
    logger.info("API startup complete")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns the current status of the API and model.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=predictor.is_loaded,
        gpu_available=predictor.device == "cuda",
        version=predictor.version
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Generate dengue risk prediction for a single location.
    
    Args:
        request: Location and forecast parameters
        
    Returns:
        Prediction with confidence intervals and risk level
    """
    try:
        lat = request.location.latitude
        lon = request.location.longitude
        
        logger.info(f"Prediction request: lat={lat}, lon={lon}, horizon={request.forecast_horizon}")
        
        result = predictor.predict(lat, lon, request.forecast_horizon)
        
        return PredictionResponse(
            location={"latitude": lat, "longitude": lon},
            predictions=result["predictions"],
            confidence=result["confidence"],
            risk_level=result["risk_level"],
            model_version=predictor.version,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest):
    """
    Generate dengue risk predictions for multiple locations.
    
    Args:
        request: List of locations and forecast parameters
        
    Returns:
        Batch predictions with aggregated statistics
    """
    try:
        locations = [(loc.latitude, loc.longitude) for loc in request.locations]
        horizon = request.forecast_horizon
        
        logger.info(f"Batch prediction request: {len(locations)} locations")
        
        results = predictor.batch_predict(locations, horizon)
        
        predictions = []
        high_risk_count = 0
        
        for loc, result in zip(request.locations, results):
            predictions.append({
                "location": {"latitude": loc.latitude, "longitude": loc.longitude},
                "predictions": result["predictions"],
                "confidence": result["confidence"],
                "risk_level": result["risk_level"],
                "model_version": predictor.version,
                "timestamp": datetime.now().isoformat()
            })
            
            if result["risk_level"] == "HIGH":
                high_risk_count += 1
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_count=len(predictions),
            high_risk_count=high_risk_count,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Quantum Dengue STPP API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }


# =============================================================================
# Metrics Endpoint for Prometheus
# =============================================================================

@app.get("/metrics")
async def metrics():
    """
    Prometheus-compatible metrics endpoint.
    
    Returns metrics in a format that Prometheus can scrape.
    """
    import psutil
    
    # Basic system metrics
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    metrics_text = f"""# HELP quantum_dengue_api_up API availability
# TYPE quantum_dengue_api_up gauge
quantum_dengue_api_up 1

# HELP quantum_dengue_cpu_percent CPU usage percentage
# TYPE quantum_dengue_cpu_percent gauge
quantum_dengue_cpu_percent {cpu_percent}

# HELP quantum_dengue_memory_percent Memory usage percentage
# TYPE quantum_dengue_memory_percent gauge
quantum_dengue_memory_percent {memory.percent}

# HELP quantum_dengue_disk_percent Disk usage percentage
# TYPE quantum_dengue_disk_percent gauge
quantum_dengue_disk_percent {disk.percent}

# HELP quantum_dengue_predictions_total Total number of predictions
# TYPE quantum_dengue_predictions_total counter
quantum_dengue_predictions_total {getattr(predictor, 'total_predictions', 0)}

# HELP quantum_dengue_predictions_high_risk Total high risk predictions
# TYPE quantum_dengue_predictions_high_risk counter
quantum_dengue_predictions_high_risk {getattr(predictor, 'high_risk_count', 0)}

# HELP quantum_dengue_model_loaded Model loaded status
# TYPE quantum_dengue_model_loaded gauge
quantum_dengue_model_loaded {1 if predictor.is_loaded else 0}

# HELP quantum_dengue_gpu_available GPU availability
# TYPE quantum_dengue_gpu_available gauge
quantum_dengue_gpu_available {1 if predictor.device == "cuda" else 0}
"""
    
    return Response(content=metrics_text, media_type="text/plain")


# =============================================================================
# Middleware for request timing and security
# =============================================================================

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time and security headers."""
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Strict-Transport-Security"] = "max-age=31536000"
    
    return response


# =============================================================================
# Exception handlers
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url),
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url),
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
