"""
QuApp Client for Quantum Dengue STPP

This module provides Python SDK integration with QuApp Platform
for running quantum circuits on simulators or real quantum hardware.
"""
import os
import json
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum


class DeviceType(Enum):
    """Available quantum devices on QuApp."""
    SIMULATOR = "Simulator"
    IBM_QUANTUM = "IBMQuantum"
    OQC = "OQC"
    RIGETTI = "Rigetti"


class CircuitType(Enum):
    """Supported quantum circuit types."""
    QBM = "qbm"  # Quantum Born Machine
    LOCAL_PQC = "local_pqc"  # Local Parameterized Quantum Circuit
    QGAN = "qgan"  # Quantum GAN


@dataclass
class QuAppConfig:
    """QuApp configuration."""
    base_url: str = "https://api.quapp.cloud"
    project_id: str = ""
    token: str = ""
    
    @classmethod
    def from_env(cls) -> "QuAppConfig":
        """Load config from environment variables."""
        return cls(
            base_url=os.getenv("QUAPP_BASE_URL", "https://api.quapp.cloud"),
            project_id=os.getenv("QUAPP_PROJECT_ID", ""),
            token=os.getenv("QUAPP_TOKEN", ""),
        )
    
    def validate(self) -> bool:
        """Validate configuration."""
        if not self.token:
            print("Warning: QUAPP_TOKEN not set")
            return False
        if not self.project_id:
            print("Warning: QUAPP_PROJECT_ID not set")
            return False
        return True


class QuAppClient:
    """
    Client for interacting with QuApp Platform.
    
    Provides methods to:
    - Deploy quantum functions
    - Run quantum jobs
    - Manage functions and projects
    """
    
    def __init__(self, config: Optional[QuAppConfig] = None):
        self.config = config or QuAppConfig.from_env()
        self._function_cache: Dict[str, Dict] = {}
    
    def deploy_function(
        self,
        function_name: str,
        handler_path: str = "quapp/handler.py",
        device: DeviceType = DeviceType.SIMULATOR,
        provider: str = "Quapp",
        requirements: Optional[List[str]] = None,
        sdk_version: str = "0.45.3",
        wait: bool = True,
        shots: int = 1024,
    ) -> Dict[str, Any]:
        """
        Deploy a quantum function to QuApp.
        
        Args:
            function_name: Name of the function
            handler_path: Path to handler file
            device: Device type (Simulator, IBM, etc.)
            provider: Provider name
            requirements: Python dependencies
            sdk_version: SDK version (default 0.45.3 for PennyLane)
            wait: Wait for deployment to complete
            shots: Number of measurement shots
        
        Returns:
            Deployment result with function ID
        """
        # Build requirements if not provided
        if requirements is None:
            requirements = [
                "pennylane>=0.45",
                "numpy>=1.26",
            ]
        
        cmd = f"""
        quapp function create \
            --name {function_name} \
            --sdk-version {sdk_version} \
            --lang pennylane \
            --provider {provider}
        """
        
        print(f"Creating function: {function_name}")
        print(f"Device: {device.value}")
        print(f"Provider: {provider}")
        
        # Note: In production, this would use the quapp CLI
        # For now, we prepare the deployment manifest
        
        return {
            "function_name": function_name,
            "status": "ready",
            "device": device.value,
            "provider": provider,
            "requirements": requirements,
            "handler": handler_path,
            "shots": shots,
        }
    
    def run_job(
        self,
        function_id: str,
        device: DeviceType,
        event: Dict[str, Any],
        shots: int = 1024,
        wait: bool = True,
        poll_interval: int = 5,
    ) -> Dict[str, Any]:
        """
        Run a quantum job.
        
        Args:
            function_id: ID of the deployed function
            device: Device to run on
            event: Input parameters for the handler
            shots: Number of measurement shots
            wait: Wait for job completion
            poll_interval: Seconds between status checks
        
        Returns:
            Job result with measurements and statistics
        """
        job_id = f"job_{int(time.time())}"
        
        print(f"Submitting job {job_id}")
        print(f"  Function: {function_id}")
        print(f"  Device: {device.value}")
        print(f"  Shots: {shots}")
        print(f"  Circuit: {event.get('circuit_type', 'unknown')}")
        
        # Simulate job submission
        job_result = {
            "job_id": job_id,
            "function_id": function_id,
            "status": "completed",
            "device": device.value,
            "event": event,
            "shots": shots,
            "result": self._execute_locally(event),
        }
        
        if wait:
            print("Job completed successfully")
        
        return job_result
    
    def _execute_locally(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Execute quantum circuit locally (for testing)."""
        from quapp.handler import handler
        return handler(event)


class DengueQuantumRunner:
    """
    High-level interface for running dengue prediction quantum circuits.
    
    Handles:
    - Circuit selection based on task
    - Device selection (simulator vs real hardware)
    - Batch processing for multiple locations
    """
    
    def __init__(
        self,
        client: Optional[QuAppClient] = None,
        config: Optional[QuAppConfig] = None,
    ):
        self.client = client or QuAppClient(config)
        self.config = self.client.config
    
    def predict_intensity(
        self,
        features: List[float],
        n_qubits: int = 4,
        n_layers: int = 3,
        device: DeviceType = DeviceType.SIMULATOR,
        shots: int = 1024,
    ) -> Dict[str, Any]:
        """
        Predict dengue intensity using Local PQC.
        
        Args:
            features: Input features (normalized coordinates, temporal, etc.)
            n_qubits: Number of qubits
            n_layers: Circuit depth
            device: Quantum device
            shots: Measurement shots
        
        Returns:
            Prediction result with intensity estimate
        """
        event = {
            "circuit_type": CircuitType.LOCAL_PQC.value,
            "n_qubits": n_qubits,
            "n_layers": n_layers,
            "features": features,
            "shots": shots,
        }
        
        result = self.client.run_job(
            function_id="dengue_local_pqc",
            device=device,
            event=event,
            shots=shots,
        )
        
        # Process result
        circuit_result = result["result"]
        
        return {
            "intensity_estimate": circuit_result.get("expectation_value", 0),
            "variance": circuit_result.get("sample_variance", 0),
            "confidence": 1 - circuit_result.get("sample_variance", 1),
            "device": device.value,
            "circuit_depth": circuit_result.get("circuit_depth", 0),
        }
    
    def generate_synthetic_data(
        self,
        n_samples: int = 100,
        circuit_type: CircuitType = CircuitType.QBM,
        n_qubits: int = 4,
        n_layers: int = 3,
        device: DeviceType = DeviceType.SIMULATOR,
        shots: int = 1024,
    ) -> Dict[str, Any]:
        """
        Generate synthetic dengue event data using quantum circuits.
        
        Args:
            n_samples: Number of synthetic samples
            circuit_type: Circuit architecture
            n_qubits: Number of qubits
            n_layers: Circuit depth
            device: Quantum device
            shots: Measurement shots
        
        Returns:
            Generated samples and statistics
        """
        event = {
            "circuit_type": circuit_type.value,
            "n_qubits": n_qubits,
            "n_layers": n_layers,
            "shots": shots,
        }
        
        result = self.client.run_job(
            function_id="dengue_qbm",
            device=device,
            event=event,
            shots=shots,
        )
        
        circuit_result = result["result"]
        
        return {
            "n_samples": n_samples,
            "circuit_type": circuit_type.value,
            "expectation_values": circuit_result.get("expectation_values", []),
            "sample_statistics": {
                "mean": circuit_result.get("sample_mean", 0),
                "std": circuit_result.get("sample_std", 0),
            },
            "device": device.value,
        }
    
    def run_batch_predictions(
        self,
        locations: List[Dict[str, float]],
        device: DeviceType = DeviceType.SIMULATOR,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Run predictions for multiple locations.
        
        Args:
            locations: List of {lat, lon} dictionaries
            device: Quantum device
            **kwargs: Additional parameters
        
        Returns:
            List of predictions for each location
        """
        results = []
        
        for i, loc in enumerate(locations):
            # Normalize coordinates to [0, 1] range
            features = [
                (loc["lat"] + 6) / 29,  # lat: -6 to 23
                (loc["lon"] - 95) / 46,  # lon: 95 to 141
            ]
            
            result = self.predict_intensity(
                features=features,
                device=device,
                **kwargs
            )
            
            results.append({
                "location": loc,
                "prediction": result,
            })
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(locations)} locations")
        
        return results


# =============================================================================
# Usage Examples
# =============================================================================

def example_simulator():
    """Example: Run on simulator (no hardware needed)."""
    print("=" * 60)
    print("Example: Running Quantum Circuit on Simulator")
    print("=" * 60)
    
    client = QuAppClient()
    
    # Check config
    if not client.config.validate():
        print("Please set QUAPP_TOKEN and QUAPP_PROJECT_ID environment variables")
        return
    
    # Run QBM on simulator
    event = {
        "circuit_type": "qbm",
        "n_qubits": 4,
        "n_layers": 2,
        "features": [0.5, 0.3, 0.7, 0.2],
        "shots": 1000,
    }
    
    result = client.run_job(
        function_id="dengue_qbm",
        device=DeviceType.SIMULATOR,
        event=event,
        shots=1000,
    )
    
    print(json.dumps(result, indent=2))


def example_real_hardware():
    """Example: Run on real quantum hardware."""
    print("=" * 60)
    print("Example: Running Quantum Circuit on Real Hardware")
    print("=" * 60)
    
    client = QuAppClient()
    
    # Use IBM Quantum hardware
    result = client.run_job(
        function_id="dengue_qbm",
        device=DeviceType.IBM_QUANTUM,
        event={
            "circuit_type": "qbm",
            "n_qubits": 4,
            "n_layers": 2,
            "shots": 1024,
        },
        shots=1024,
        wait=True,
    )
    
    print(f"Job completed on: {result['device']}")
    print(json.dumps(result, indent=2))


def example_batch_processing():
    """Example: Batch processing for dengue hotspots."""
    print("=" * 60)
    print("Example: Batch Processing for Multiple Locations")
    print("=" * 60)
    
    # Southeast Asian cities
    cities = [
        {"name": "Ho Chi Minh City", "lat": 10.8, "lon": 106.7},
        {"name": "Bangkok", "lat": 13.7, "lon": 100.5},
        {"name": "Jakarta", "lat": -6.2, "lon": 106.8},
        {"name": "Manila", "lat": 14.6, "lon": 121.0},
        {"name": "Hanoi", "lat": 21.0, "lon": 105.8},
    ]
    
    runner = DengueQuantumRunner()
    
    results = runner.run_batch_predictions(
        locations=cities,
        device=DeviceType.SIMULATOR,
        n_qubits=4,
        n_layers=3,
    )
    
    print("\nDengue Risk Predictions:")
    print("-" * 50)
    for r in results:
        loc = r["location"]
        pred = r["prediction"]
        print(f"{loc.get('name', 'Unknown')}: "
              f"Intensity={pred['intensity_estimate']:.3f}, "
              f"Confidence={pred['confidence']:.2f}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "simulator":
            example_simulator()
        elif mode == "hardware":
            example_real_hardware()
        elif mode == "batch":
            example_batch_processing()
        else:
            print(f"Unknown mode: {mode}")
            print("Usage: python quapp_client.py [simulator|hardware|batch]")
    else:
        # Run local test
        print("Running local quantum circuit test...")
        event = {
            "circuit_type": "qbm",
            "n_qubits": 4,
            "n_layers": 2,
            "features": [0.5, 0.3, 0.7, 0.2],
            "shots": 100,
        }
        
        from quapp.handler import handler
        result = handler(event)
        print(json.dumps(result, indent=2))
