"""
FastAPI endpoints for Quantum Dengue STPP predictions.

Provides REST API for:
- Health check
- Model predictions
- Batch forecasting
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from typing import List, Optional, Dict, Any
from datetime import datetime
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Quantum Dengue STPP API",
    description="Quantum-Enhanced Spatio-Temporal Point Process for Dengue Prediction",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Request/Response Models
# =============================================================================

class LocationInput(BaseModel):
    """Geographic location input."""
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    
    @validator('latitude', 'longitude')
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
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
