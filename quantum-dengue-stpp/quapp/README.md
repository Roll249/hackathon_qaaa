# QuApp Deployment Guide

Guide to deploy and run Quantum Dengue STPP on [QuApp Platform](https://quapp.cloud/).

## Overview

QuApp provides cloud-based quantum computing with:
- **Simulators**: Test circuits without hardware access
- **Real Hardware**: IBM Quantum, OQC, Rigetti
- **SDK Support**: Qiskit, PennyLane, Cirq

## Quick Start

### 1. Install QuApp CLI

```bash
pip install quapp
```

Or download standalone binary:
```bash
# Linux
wget https://gitlab.com/quapp/quapp-cli/-/pipelines/latest/artifacts -O quapp
chmod +x quapp
```

### 2. Configure Authentication

```bash
# Set environment variables
export QUAPP_TOKEN="your-api-token"
export QUAPP_PROJECT_ID="your-project-id"

# Or login via CLI
quapp login
```

Get your credentials from [QuApp Dashboard](https://quapp.cloud/).

### 3. Deploy Quantum Function

```bash
cd quantum-dengue-stpp

# Deploy with one command
quapp deploy \
  --project quantum-dengue \
  --function dengue-qbm \
  --handler quapp/handler.py \
  --device Simulator \
  --shots 1024 \
  --requirements requirements.txt \
  --wait
```

### 4. Run Quantum Job

```bash
# Run QBM circuit
quapp job run \
  --function dengue-qbm \
  --device Simulator \
  --shots 1024 \
  --input circuit_type=qbm,n_qubits=4,n_layers=3 \
  --wait

# Run Local PQC circuit
quapp job run \
  --function dengue-local-pqc \
  --device IBMQuantum \
  --shots 2048 \
  --input circuit_type=local_pqc,n_qubits=4,n_layers=3 \
  --wait
```

## Supported Devices

| Device | Type | Qubits | Notes |
|--------|------|--------|-------|
| Simulator | Local | Unlimited | Fast testing |
| qiskit.simulator | Aer | Unlimited | High fidelity |
| ibmq_quito | Superconducting | 5 | Real hardware |
| ibmq_belem | Superconducting | 5 | Real hardware |
| ibmq_manila | Superconducting | 5 | Real hardware |
| oqc.lucy | Superconducting | 8 | UK-based |
| Aspen-11 | Superconducting | 32 | Rigetti |

## Python SDK Usage

```python
from quapp.quapp_client import QuAppClient, DeviceType, DengueQuantumRunner

# Initialize client
client = QuAppClient()

# Deploy function
client.deploy_function(
    function_name="dengue-qbm",
    handler_path="quapp/handler.py",
    device=DeviceType.SIMULATOR,
)

# Run job
result = client.run_job(
    function_id="dengue-qbm",
    device=DeviceType.IBM_QUANTUM,
    event={
        "circuit_type": "qbm",
        "n_qubits": 4,
        "n_layers": 3,
        "features": [0.5, 0.3, 0.7, 0.2],
        "shots": 1024,
    },
    shots=1024,
    wait=True,
)

print(result)
```

## High-Level Interface

```python
from quapp.quapp_client import DengueQuantumRunner, DeviceType

runner = DengueQuantumRunner()

# Single prediction
result = runner.predict_intensity(
    features=[0.5, 0.7],  # Normalized lat, lon
    n_qubits=4,
    n_layers=3,
    device=DeviceType.SIMULATOR,
    shots=1024,
)
print(f"Intensity: {result['intensity_estimate']}")

# Batch predictions
cities = [
    {"lat": 10.8, "lon": 106.7},  # Ho Chi Minh City
    {"lat": 13.7, "lon": 100.5},  # Bangkok
    {"lat": -6.2, "lon": 106.8},  # Jakarta
]

results = runner.run_batch_predictions(
    locations=cities,
    device=DeviceType.SIMULATOR,
)
```

## Circuit Types

### 1. Quantum Born Machine (QBM)
```python
{
    "circuit_type": "qbm",
    "n_qubits": 4,
    "n_layers": 3,
    "features": [0.5, 0.3, 0.7, 0.2],
    "shots": 1024,
}
```
Learns probability distribution of dengue events.

### 2. Local PQC
```python
{
    "circuit_type": "local_pqc",
    "n_qubits": 4,
    "n_layers": 3,
    "features": [0.5, 0.7],  # lat, lon
    "shots": 1024,
}
```
Spatial clustering-based intensity prediction.

### 3. QGAN
```python
{
    "circuit_type": "qgan",
    "n_qubits": 4,
    "n_layers": 3,
    "shots": 1024,
}
```
Generative adversarial training.

## Deployment Options

### Option 1: CLI One-Liner
```bash
quapp deploy \
  --project quantum-dengue \
  --function dengue-qbm \
  --handler quapp/handler.py \
  --device Simulator \
  --shots 1024 \
  --wait
```

### Option 2: Step-by-Step
```bash
# 1. Create function
quapp function create \
  --name dengue-qbm \
  --sdk-version 0.45.3 \
  --lang pennylane

# 2. Upload version
quapp function version <function-id> \
  --description "v1.0" \
  --files quapp/handler.py

# 3. Deploy
quapp function deploy <function-id>

# 4. Run job
quapp job run \
  --function <function-id> \
  --device Simulator \
  --shots 1024 \
  --wait
```

### Option 3: Python SDK
```python
from quapp.api.client import QuappClient

client = QuAppClient(
    base_url="https://api.quapp.cloud",
    token="your-token",
    project_id="your-project",
)

# Programmatic deployment and execution
job = run_job(
    client,
    function_id="f123",
    provider="Quapp",
    device="Simulator",
    shots=1024,
)
```

## Monitoring Jobs

```bash
# List jobs
quapp job list

# Get job status
quapp job status <job-id>

# Get job result
quapp job result <job-id>
```

## Cost Estimation

| Device | Cost Unit | Est. Cost (1K shots) |
|--------|-----------|---------------------|
| Simulator | Free | $0.00 |
| qiskit.simulator | Per shot | $0.001 |
| ibmq_quito | Per shot | $0.01 |
| ibmq_belem | Per shot | $0.01 |
| oqc.lucy | Per shot | $0.02 |

## Troubleshooting

### Authentication Error
```bash
# Check token
echo $QUAPP_TOKEN

# Re-login
quapp logout && quapp login
```

### Device Not Available
```bash
# List available devices
quapp device list

# Check device status
quapp device status ibmq_quito
```

### Job Timeout
```bash
# Increase timeout
quapp job run --function f123 --device Simulator --timeout 600
```

## Environment Variables

```bash
# Required
export QUAPP_TOKEN="your-api-token"
export QUAPP_PROJECT_ID="your-project-id"

# Optional
export QUAPP_BASE_URL="https://api.quapp.cloud"  # Default
export QUAPP_TIMEOUT=300  # Job timeout in seconds
```

## Resources

- [QuApp Documentation](https://docs.quapp.cloud/)
- [QuApp Dashboard](https://quapp.cloud/)
- [GitHub](https://github.com/quapp)
- [Support](https://quapp.cloud/support)
