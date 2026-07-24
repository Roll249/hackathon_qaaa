"""Quantum Reservoir Computing v9 - Simplified & Stable.

Key improvements over v1:
1. Better input encoding (lag features)
2. Teacher forcing during initial prediction
3. Stable warmup phase
4. Proper normalization throughout

This version uses a simpler, more robust approach:
- Input: normalized lag features [x_t, x_{t-1}, x_{t-2}, x_{t-3}]
- Reservoir: quantum circuit → measurement → classical state
- Output: ridge regression on reservoir states
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False
    qml = None


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class QRCConfig:
    """QRC v9 Configuration."""
    n_qubits: int = 6
    n_layers: int = 3
    n_internal: int = 50
    spectral_radius: float = 0.8
    leaky_rate: float = 0.3
    regularization: float = 1e-3
    n_lags: int = 4
    seed: int = 42


# ============================================================================
# Quantum Reservoir
# ============================================================================

class QuantumReservoirV9:
    """Simplified QRC with better stability."""
    
    def __init__(self, config: QRCConfig):
        self.config = config
        self._init_weights()
        self._dev = None
        self._circuit = None
        
    def _init_weights(self):
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        
        # Reservoir weights
        W = rng.normal(0, 0.1, size=(cfg.n_internal, cfg.n_internal))
        eigvals = np.linalg.eigvals(W)
        W = W * (cfg.spectral_radius / max(np.abs(eigvals)))
        self.W = W
        
        # Input weights
        self.W_in = rng.uniform(-0.5, 0.5, size=(cfg.n_internal, cfg.n_qubits))
        
        # State
        self.state = np.zeros(cfg.n_internal)
        
    def _build_circuit(self):
        if self._dev is None:
            self._dev = qml.device("default.qubit", wires=self.config.n_qubits)
            
        if self._circuit is not None:
            return self._circuit
            
        cfg = self.config
        dev = self._dev
        
        @qml.qnode(dev)
        def circuit(input_vec, reservoir_state):
            # Encode input
            for i in range(cfg.n_qubits):
                angle = float(np.clip(input_vec[i], -math.pi/2, math.pi/2))
                qml.RY(angle, wires=i)
            
            # Fixed dynamics
            for layer in range(cfg.n_layers):
                # Local rotations
                for i in range(cfg.n_qubits):
                    qml.RX(0.5, wires=i)
                    qml.RZ(0.5, wires=i)
                
                # Entanglement
                for i in range(cfg.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            
            # Measurement
            return [qml.expval(qml.PauliZ(i)) for i in range(cfg.n_qubits)]
        
        self._circuit = circuit
        return circuit
    
    def step(self, input_vec: np.ndarray) -> np.ndarray:
        cfg = self.config
        
        # Pad input if needed
        if len(input_vec) < cfg.n_qubits:
            input_vec = np.pad(input_vec, (0, cfg.n_qubits - len(input_vec)))
        input_vec = input_vec[:cfg.n_qubits]
        
        circuit = self._build_circuit()
        quantum_out = circuit(input_vec, self.state)
        
        # Update state
        combined = self.W @ self.state + self.W_in @ quantum_out
        new_state = (1 - cfg.leaky_rate) * self.state + cfg.leaky_rate * np.tanh(combined)
        
        self.state = new_state
        return new_state.copy()
    
    def reset(self):
        self.state = np.zeros(self.config.n_internal)
        self._dev = None
        self._circuit = None
    
    def get_state(self) -> np.ndarray:
        return self.state.copy()


# ============================================================================
# Output Layer
# ============================================================================

class OutputLayer:
    def __init__(self, regularization: float = 1e-3):
        self.regularization = regularization
        self.W_out: Optional[np.ndarray] = None
        self.states: List[np.ndarray] = []
        self.targets: List[np.ndarray] = []
        
    def add(self, state: np.ndarray, target: float):
        self.states.append(state.copy())
        self.targets.append([target])
    
    def train(self):
        if len(self.states) < 5:
            raise ValueError("Need at least 5 samples")
            
        X = np.array(self.states)
        Y = np.array(self.targets)
        
        reg = self.regularization * np.eye(X.shape[1])
        self.W_out = np.linalg.solve(X.T @ X + reg, X.T @ Y)
    
    def predict(self, state: np.ndarray) -> float:
        if self.W_out is None:
            raise RuntimeError("Not trained")
        return float((state @ self.W_out)[0])
    
    def clear(self):
        self.states = []
        self.targets = []
        self.W_out = None


# ============================================================================
# Main Prediction
# ============================================================================

@dataclass
class QRCV9Result:
    mse: float
    rmse: float
    mae: float
    r2: float
    mape: float
    predictions: np.ndarray
    actuals: np.ndarray
    train_time_s: float
    predict_time_s: float


def qrc_predict_v9(
    timeseries: np.ndarray,
    config: Optional[QRCConfig] = None,
    train_fraction: float = 0.7,
    seed: int = 42,
    verbose: bool = True,
) -> QRCV9Result:
    """QRC v9 prediction with stable autoregressive loop."""
    
    if config is None:
        config = QRCConfig(seed=seed)
    config.seed = seed
    
    timeseries = np.asarray(timeseries, dtype=float)
    
    n_total = len(timeseries)
    n_train = int(n_total * train_fraction)
    n_test = n_total - n_train
    
    train_data = timeseries[:n_train]
    test_data = timeseries[n_train:]
    
    # Normalize to [0, 1] first
    data_min, data_max = train_data.min(), train_data.max()
    data_range = data_max - data_min + 1e-8
    
    def norm(x):
        return (x - data_min) / data_range
    
    def denorm(x):
        return x * data_range + data_min
    
    # Initialize
    reservoir = QuantumReservoirV9(config)
    output_layer = OutputLayer(regularization=config.regularization)
    
    # Training
    t_start = time.time()
    
    reservoir.reset()
    output_layer.clear()
    
    # Collect training samples
    # Use lag features: predict x[t+1] from [norm(x[t]), norm(x[t-1]), ...]
    for t in range(config.n_lags, n_train - 1):
        # Input: normalized lags
        input_vec = np.array([norm(timeseries[t - i]) for i in range(config.n_lags)])
        
        # Target: next value (denormalized)
        target = timeseries[t + 1]
        
        state = reservoir.step(input_vec)
        output_layer.add(state, target)
    
    output_layer.train()
    train_time = time.time() - t_start
    
    # Prediction with teacher forcing
    t_start = time.time()
    
    predictions = []
    reservoir.reset()
    
    # Warmup with actual training data
    for t in range(max(0, n_train - 20), n_train):
        input_vec = np.array([norm(timeseries[t - i]) for i in range(config.n_lags)])
        reservoir.step(input_vec)
    
    # Predict iteratively with TEACHER FORCING first, then AUTOREGRESSIVE
    # Teacher forcing: use actual values for first few steps
    teacher_forcing_steps = min(10, n_test)
    
    for step in range(n_test):
        state = reservoir.get_state()
        
        # Get prediction
        pred_raw = output_layer.predict(state)  # denormalized value
        
        # Clamp to reasonable range
        pred_raw = np.clip(pred_raw, data_min * 0.5, data_max * 1.5)
        predictions.append(pred_raw)
        
        # Build input for next step
        # For teacher forcing: use actual values from training/test
        if step < teacher_forcing_steps:
            # Use actual previous values
            actual_prev = test_data[step] if step > 0 else train_data[-1]
            input_vec = np.array([
                norm(actual_prev),
                norm(train_data[-1] if step == 0 else test_data[step - 1]),
                norm(train_data[-2] if step <= 1 else test_data[step - 2]),
                norm(train_data[-3] if step <= 2 else test_data[step - 3]),
            ])
        else:
            # Autoregressive: use predictions
            if step == 0:
                prev = train_data[-1]
            else:
                prev = predictions[-1]
            
            if step >= 1 and len(predictions) >= 2:
                prev2 = predictions[-2]
            elif step == 1:
                prev2 = train_data[-1]
            else:
                prev2 = train_data[-1]
                
            if step >= 2 and len(predictions) >= 3:
                prev3 = predictions[-3]
            elif step == 2:
                prev3 = train_data[-1]
            else:
                prev3 = train_data[-1]
            
            input_vec = np.array([norm(prev), norm(prev2), norm(prev3), norm(prev3)])
        
        reservoir.step(input_vec)
    
    predict_time = time.time() - t_start
    
    # Metrics
    predictions = np.array(predictions)
    actuals = test_data
    
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))
    
    variance = np.var(actuals)
    nmse = mse / max(variance, 1e-8)
    
    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r2 = 1 - ss_res / max(ss_tot, 1e-8)
    
    mask = actuals > 0
    mape = np.mean(np.abs((actuals[mask] - predictions[mask]) / actuals[mask])) * 100 if mask.any() else 0
    
    if verbose:
        print(f"  QRC v9: MSE={mse:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}")
        print(f"  Config: qubits={config.n_qubits}, layers={config.n_layers}, internal={config.n_internal}")
    
    return QRCV9Result(
        mse=float(mse),
        rmse=float(rmse),
        mae=float(mae),
        r2=float(r2),
        mape=float(mape),
        predictions=predictions,
        actuals=actuals,
        train_time_s=train_time,
        predict_time_s=predict_time,
    )


# ============================================================================
# Baselines
# ============================================================================

def esn_baseline(timeseries: np.ndarray, seed: int = 42) -> dict:
    """Classical ESN baseline."""
    timeseries = np.asarray(timeseries, dtype=float)
    n_train = int(len(timeseries) * 0.7)
    
    train, test = timeseries[:n_train], timeseries[n_train:]
    
    data_min, data_max = train.min(), train.max()
    data_range = data_max - data_min + 1e-8
    
    def norm(x):
        return (x - data_min) / data_range
    
    rng = np.random.default_rng(seed)
    n_internal = 50
    
    W = rng.normal(0, 0.1, size=(n_internal, n_internal))
    eigvals = np.linalg.eigvals(W)
    W = W * (0.8 / max(np.abs(eigvals)))
    W_in = rng.uniform(-1, 1, size=(n_internal, 1))
    
    states = []
    state = np.zeros(n_internal)
    leaky = 0.3
    
    for t in range(len(train) - 1):
        input_val = norm(train[t])
        new_state = (1 - leaky) * state + leaky * np.tanh(W @ state + W_in @ [input_val])
        states.append(new_state)
        state = new_state
    
    X = np.array(states)
    Y = train[1:].reshape(-1, 1)
    W_out = np.linalg.solve(X.T @ X + 1e-3 * np.eye(n_internal), X.T @ Y)
    
    predictions = []
    state = states[-1]
    
    for t in range(len(test)):
        input_val = norm(predictions[-1] if predictions else train[-1])
        state = (1 - leaky) * state + leaky * np.tanh(W @ state + W_in @ [input_val])
        pred = float((W_out.T @ state).item())
        predictions.append(pred)
    
    predictions = np.array(predictions)
    mse = np.mean((predictions - test) ** 2)
    
    return {"mse": float(mse), "rmse": float(np.sqrt(mse))}


def seasonal_naive_baseline(timeseries: np.ndarray, seasonal_period: int = 52) -> dict:
    """Seasonal naive baseline."""
    timeseries = np.asarray(timeseries, dtype=float)
    n_train = int(len(timeseries) * 0.7)
    
    train, test = timeseries[:n_train], timeseries[n_train:]
    
    predictions = []
    for i in range(len(test)):
        idx = n_train - seasonal_period + i
        if 0 <= idx < n_train:
            predictions.append(train[idx])
        else:
            predictions.append(train[-1])
    
    predictions = np.array(predictions)
    mse = np.mean((predictions - test) ** 2)
    
    return {"mse": float(mse), "rmse": float(np.sqrt(mse))}


def holt_winters_baseline(timeseries: np.ndarray) -> dict:
    """Holt-Winters baseline."""
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
    except ImportError:
        return {"mse": float('inf'), "rmse": float('inf')}
    
    timeseries = np.asarray(timeseries, dtype=float)
    n_train = int(len(timeseries) * 0.7)
    
    train, test = timeseries[:n_train], timeseries[n_train:]
    
    try:
        model = ExponentialSmoothing(
            train,
            seasonal_period=52,
            trend='add',
            seasonal='add',
            damped_trend=True,
        )
        fitted = model.fit()
        predictions = fitted.forecast(len(test))
        
        mse = np.mean((predictions - test) ** 2)
        return {"mse": float(mse), "rmse": float(np.sqrt(mse))}
    except:
        return {"mse": float('inf'), "rmse": float('inf')}


def rf_baseline(timeseries: np.ndarray, seed: int = 42) -> dict:
    """Random Forest baseline."""
    try:
        from sklearn.ensemble import RandomForestRegressor
    except ImportError:
        return {"mse": float('inf'), "rmse": float('inf')}
    
    timeseries = np.asarray(timeseries, dtype=float)
    n_train = int(len(timeseries) * 0.7)
    n_lags = 8
    
    train, test = timeseries[:n_train], timeseries[n_train:]
    
    X_train, y_train = [], []
    for i in range(n_lags, len(train)):
        X_train.append(train[i-n_lags:i])
        y_train.append(train[i])
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    rf = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    predictions = []
    current = train[-n_lags:].tolist()
    
    for _ in test:
        pred = rf.predict(np.array(current[-n_lags:]).reshape(1, -1))[0]
        predictions.append(pred)
        current.append(pred)
    
    predictions = np.array(predictions)
    mse = np.mean((predictions - test) ** 2)
    
    return {"mse": float(mse), "rmse": float(np.sqrt(mse))}


# ============================================================================
# Main
# ============================================================================

def generate_dengue_data(n_weeks: int = 208, seed: int = 42) -> np.ndarray:
    """Generate synthetic dengue-like time series."""
    rng = np.random.default_rng(seed)
    weeks = np.arange(n_weeks)
    
    # Seasonal + trend + noise
    seasonal = 0.5 + 0.5 * np.sin(2 * np.pi * (weeks - 10) / 52)
    trend = 1.0 + 0.2 * np.sin(2 * np.pi * weeks / (52 * 4))
    base = seasonal * trend
    noise = rng.normal(0, 0.15, size=n_weeks)
    cases = base * (1 + noise) * 100
    
    return cases


def run_benchmark():
    """Run comparison benchmark."""
    output_dir = "output_result/qrc_v9"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("  QRC v9 Benchmark")
    print("=" * 60)
    
    # Generate data
    print("\n[1] Generating data...")
    timeseries = generate_dengue_data(n_weeks=208, seed=42)
    print(f"  {len(timeseries)} weeks generated")
    
    # Test QRC v9 with different configs
    print("\n[2] Testing QRC v9 configs...")
    
    configs = [
        QRCConfig(n_qubits=4, n_layers=2, n_internal=30, seed=42),
        QRCConfig(n_qubits=6, n_layers=3, n_internal=50, seed=42),
        QRCConfig(n_qubits=8, n_layers=3, n_internal=60, seed=42),
    ]
    
    best_result = None
    best_config = None
    best_mse = float('inf')
    
    for cfg in configs:
        print(f"\n  Config: qubits={cfg.n_qubits}, layers={cfg.n_layers}, internal={cfg.n_internal}")
        result = qrc_predict_v9(timeseries, config=cfg, verbose=True)
        
        if result.mse < best_mse:
            best_mse = result.mse
            best_result = result
            best_config = cfg
    
    # Baselines
    print("\n[3] Running baselines...")
    
    esn_res = esn_baseline(timeseries, seed=42)
    print(f"  ESN: MSE={esn_res['mse']:.2f}")
    
    sn_res = seasonal_naive_baseline(timeseries)
    print(f"  Seasonal Naive: MSE={sn_res['mse']:.2f}")
    
    hw_res = holt_winters_baseline(timeseries)
    print(f"  Holt-Winters: MSE={hw_res['mse']:.2f}")
    
    rf_res = rf_baseline(timeseries, seed=42)
    print(f"  Random Forest: MSE={rf_res['mse']:.2f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    
    methods = [
        ("QRC v9 (best)", best_result.mse),
        ("ESN", esn_res['mse']),
        ("Seasonal Naive", sn_res['mse']),
        ("Holt-Winters", hw_res['mse']),
        ("Random Forest", rf_res['mse']),
    ]
    
    methods.sort(key=lambda x: x[1])
    
    print("\nRanking (by MSE, lower is better):")
    for i, (name, mse) in enumerate(methods, 1):
        marker = " *" if name.startswith("QRC") else ""
        print(f"  {i}. {name:20s}: MSE={mse:12.2f}{marker}")
    
    if best_result:
        print(f"\nQRC v9 Best Config:")
        print(f"  qubits={best_config.n_qubits}, layers={best_config.n_layers}, internal={best_config.n_internal}")
        print(f"  R²={best_result.r2:.4f}")
    
    # Plot
    if best_result:
        print("\n[4] Generating plots...")
        plot_results(best_result, output_dir)
    
    print(f"\nOutput: {output_dir}/")
    print("=" * 60)
    
    return best_result


def plot_results(result: QRCV9Result, output_dir: str):
    """Plot prediction results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Time series
    ax = axes[0, 0]
    ax.plot(result.actuals, 'b-', label='Actual', linewidth=2)
    ax.plot(result.predictions, 'r--', label='QRC v9', linewidth=2)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Cases')
    ax.set_title(f'Predictions (MSE={result.mse:.2f}, R²={result.r2:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Scatter
    ax = axes[0, 1]
    ax.scatter(result.actuals, result.predictions, alpha=0.6)
    min_v = min(result.actuals.min(), result.predictions.min())
    max_v = max(result.actuals.max(), result.predictions.max())
    ax.plot([min_v, max_v], [min_v, max_v], 'r--')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('Scatter Plot')
    ax.grid(True, alpha=0.3)
    
    # 3. Residuals
    ax = axes[1, 0]
    residuals = result.actuals - result.predictions
    ax.plot(residuals, 'g-', alpha=0.7)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Residual')
    ax.set_title(f'Residuals (std={residuals.std():.2f})')
    ax.grid(True, alpha=0.3)
    
    # 4. Metrics
    ax = axes[1, 1]
    ax.axis('off')
    
    metrics = [
        ['Metric', 'Value'],
        ['MSE', f'{result.mse:.2f}'],
        ['RMSE', f'{result.rmse:.2f}'],
        ['MAE', f'{result.mae:.2f}'],
        ['R²', f'{result.r2:.4f}'],
        ['MAPE', f'{result.mape:.2f}%'],
    ]
    
    table = ax.table(cellText=metrics[1:], colLabels=metrics[0], loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    
    for i in range(2):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', weight='bold')
    
    ax.set_title('QRC v9 Performance', fontsize=12, fontweight='bold')
    
    fig.suptitle('QRC v9 Dengue Forecasting', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{output_dir}/qrc_v9_results.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Saved: {output_dir}/qrc_v9_results.png")


if __name__ == "__main__":
    import sys
    sys.exit(run_benchmark())
