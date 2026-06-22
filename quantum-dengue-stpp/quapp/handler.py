"""
QuApp Quantum Handler for Dengue STPP Prediction

This handler runs quantum circuits for dengue prediction via QuApp Platform.
Supports both simulators and real quantum hardware.
"""
import json
import numpy as np
from typing import Dict, Any, Optional


def handler(
    event: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    QuApp handler function for quantum circuit execution.
    
    Args:
        event: Input parameters containing:
            - circuit_type: "qbm" | "local_pqc" | "qgan"
            - n_qubits: Number of qubits
            - n_layers: Number of circuit layers
            - features: Input features for circuit
            - shots: Number of measurement shots
        context: QuApp execution context (optional)
    
    Returns:
        Dictionary with:
            - results: Circuit measurement results
            - counts: Measurement counts
            - expectation_values: Expectation values
            - device_used: Device that executed the circuit
    """
    try:
        # Extract parameters
        circuit_type = event.get("circuit_type", "qbm")
        n_qubits = event.get("n_qubits", 4)
        n_layers = event.get("n_layers", 3)
        features = event.get("features", None)
        shots = event.get("shots", 1024)
        
        # Device info from QuApp
        device_info = event.get("_quapp_device", "Simulator")
        
        # Execute quantum circuit
        if circuit_type == "qbm":
            results = run_qbm_circuit(n_qubits, n_layers, features, shots)
        elif circuit_type == "local_pqc":
            results = run_local_pqc_circuit(n_qubits, n_layers, features, shots)
        elif circuit_type == "qgan":
            results = run_qgan_circuit(n_qubits, n_layers, features, shots)
        else:
            raise ValueError(f"Unknown circuit type: {circuit_type}")
        
        return {
            "status": "success",
            "circuit_type": circuit_type,
            "results": results,
            "device_used": device_info,
            "n_qubits": n_qubits,
            "n_layers": n_layers,
            "shots": shots,
        }
    
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "circuit_type": circuit_type,
        }


def run_qbm_circuit(
    n_qubits: int,
    n_layers: int,
    features: Optional[np.ndarray],
    shots: int
) -> Dict[str, Any]:
    """
    Run Quantum Born Machine circuit.
    
    The QBM learns the probability distribution of dengue events
    using parameterized quantum circuits.
    """
    import pennylane as qml
    from pennylane import numpy as pnp
    
    # Set random seed for reproducibility
    pnp.random.seed(42)
    
    # Initialize weights
    weights = pnp.random.randn(n_layers, n_qubits, 3, requires_grad=True)
    
    # Create two circuits: one for expectation values, one for samples
    dev_expval = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(dev_expval, interface="autograd")
    def expval_circuit(weights, feature_vector=None):
        # Encode features if provided
        if feature_vector is not None:
            for i in range(min(len(feature_vector), n_qubits)):
                qml.RX(feature_vector[i], wires=i)
        
        # Variational layers
        for layer in range(n_layers):
            # Rotation gates
            for qubit in range(n_qubits):
                qml.RY(weights[layer, qubit, 0], wires=qubit)
                qml.RZ(weights[layer, qubit, 1], wires=qubit)
                qml.RX(weights[layer, qubit, 2], wires=qubit)
            
            # Entangling gates
            for qubit in range(n_qubits - 1):
                qml.CNOT(wires=[qubit, qubit + 1])
            if n_qubits > 2:
                qml.CNOT(wires=[n_qubits - 1, 0])
        
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    # Create sample device
    dev_sample = qml.device("default.qubit", wires=n_qubits, shots=shots)
    
    @qml.qnode(dev_sample, interface="autograd")
    def sample_circuit(weights, feature_vector=None):
        if feature_vector is not None:
            for i in range(min(len(feature_vector), n_qubits)):
                qml.RX(feature_vector[i], wires=i)
        
        for layer in range(n_layers):
            for qubit in range(n_qubits):
                qml.RY(weights[layer, qubit, 0], wires=qubit)
                qml.RZ(weights[layer, qubit, 1], wires=qubit)
                qml.RX(weights[layer, qubit, 2], wires=qubit)
            for qubit in range(n_qubits - 1):
                qml.CNOT(wires=[qubit, qubit + 1])
            if n_qubits > 2:
                qml.CNOT(wires=[n_qubits - 1, 0])
        
        return qml.sample()
    
    # Run circuits
    if features is not None:
        features = pnp.array(features)

    expectation_values = expval_circuit(weights, feature_vector=features)
    exp_values = [float(qml.math.detach(ev)) for ev in expectation_values]
    
    # Get measurement samples
    samples = sample_circuit(weights, feature_vector=features)
    if hasattr(samples, 'numpy'):
        samples = samples.numpy()
    
    return {
        "expectation_values": exp_values,
        "sample_mean": float(np.mean(samples)),
        "sample_std": float(np.std(samples)),
    }


def run_local_pqc_circuit(
    n_qubits: int,
    n_layers: int,
    features: Optional[np.ndarray],
    shots: int
) -> Dict[str, Any]:
    """
    Run Local Parameterized Quantum Circuit (Local PQC).
    
    Used for spatial clustering-based quantum augmentation.
    """
    import pennylane as qml
    from pennylane import numpy as pnp
    
    pnp.random.seed(42)
    
    # Create device
    dev = qml.device("default.qubit", wires=n_qubits, shots=shots)
    
    @qml.qnode(dev, interface="autograd")
    def circuit(weights, x):
        # Feature encoding
        for i in range(min(len(x), n_qubits)):
            qml.Hadamard(wires=i)
            qml.RZ(x[i], wires=i)
        
        # StronglyEntanglingLayers
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RY(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)
            
            # Multi-qubit gates
            for i in range(n_qubits - 1):
                qml.CZ(wires=[i, i + 1])
        
        return qml.expval(qml.PauliZ(0))
    
    weights = pnp.random.randn(n_layers, n_qubits, 2, requires_grad=True)
    
    if features is not None:
        features = pnp.array(features)
    else:
        features = pnp.zeros(n_qubits)

    exp_val = circuit(weights, x=features)
    exp_value = float(qml.math.detach(exp_val))
    
    samples = circuit(weights, x=features)
    if samples is not None and hasattr(samples, 'numpy'):
        samples = samples.numpy()
    
    return {
        "expectation_value": exp_value,
        "sample_variance": float(np.var(samples)) if isinstance(samples, np.ndarray) else 0.0,
        "circuit_depth": n_layers * n_qubits,
    }


def run_qgan_circuit(
    n_qubits: int,
    n_layers: int,
    features: Optional[np.ndarray],
    shots: int
) -> Dict[str, Any]:
    """
    Run QGAN-style quantum circuit for generative modeling.
    """
    import pennylane as qml
    from pennylane import numpy as pnp
    
    pnp.random.seed(42)
    
    dev = qml.device("default.qubit", wires=n_qubits, shots=shots)
    
    @qml.qnode(dev, interface="autograd")
    def generator_circuit(weights):
        # Initial superposition
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        
        # Generator layers
        for layer in range(n_layers):
            # Single qubit rotations
            for i in range(n_qubits):
                qml.RY(weights[layer, i, 0], wires=i)
                qml.RX(weights[layer, i, 1], wires=i)
            
            # Entanglement pattern
            for i in range(0, n_qubits - 1, 2):
                qml.CNOT(wires=[i, i + 1])
            for i in range(1, n_qubits - 1, 2):
                qml.CNOT(wires=[i, i + 1])
        
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    @qml.qnode(dev, interface="autograd")
    def discriminator_circuit(weights, generator_output):
        # Encode generator output
        for i in range(n_qubits):
            qml.RX(generator_output[i], wires=i)
            qml.RY(generator_output[i] * 0.5, wires=i)
        
        # Discriminator layers
        for layer in range(2):
            for i in range(n_qubits):
                qml.RZ(weights[layer, i], wires=i)
        
        return qml.expval(qml.PauliZ(0))
    
    gen_weights = pnp.random.randn(n_layers, n_qubits, 2, requires_grad=True)
    disc_weights = pnp.random.randn(2, n_qubits, requires_grad=True)

    # Run generator
    gen_output = generator_circuit(gen_weights)
    gen_values = [float(qml.math.detach(g)) for g in gen_output]
    
    # Run discriminator
    disc_output = discriminator_circuit(disc_weights, gen_output)
    disc_value = float(qml.math.detach(disc_output))
    
    return {
        "generator_output": gen_values,
        "discriminator_score": disc_value,
        "adversarial_loss_proxy": float(abs(disc_value)),
    }


if __name__ == "__main__":
    # Local testing
    test_event = {
        "circuit_type": "qbm",
        "n_qubits": 4,
        "n_layers": 2,
        "features": [0.5, 0.3, 0.7, 0.2],
        "shots": 1000,
    }
    
    result = handler(test_event)
    print(json.dumps(result, indent=2))
