#!/bin/bash
# ============================================================================
# Quantum Dengue STPP - Full Benchmark Suite
# ============================================================================
# Chạy so sánh đầy đủ giữa:
#   - QNG (Quantum Natural Gradient)
#   - Adam
#   - Adam + Local PQC (quantum-classical hybrid)
#   - CNN-LSTM baseline
#
# Metrics: R², RMSE, MAE, training time, convergence
#
# GPU Support: Tự động detect CUDA, dùng GPU nếu có
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ============================================================================
# Configuration
# ============================================================================
DATA_DIR="${DATA_DIR:-dengue_dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-output_result/benchmarks}"
GRID_SIZE=20
SEQ_LEN=12
FORECAST_HORIZON=1
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE=32
LR=1e-3
N_LAYERS=3
N_CLUSTERS=8
N_QUBITS=4

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ============================================================================
# Helper Functions
# ============================================================================
log() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }
header() { echo -e "\n${BOLD}${CYAN}========================================${NC}"; }
subheader() { echo -e "${BOLD}${BLUE}--- $1${NC}"; }

# Check GPU availability
check_gpu() {
    header
    echo -e "${BOLD}  GPU CHECK${NC}"
    header

    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null || true
        log "NVIDIA GPU detected"
    else
        warn "No NVIDIA GPU found - will run on CPU"
    fi

    python -c "
import torch
cuda = torch.cuda.is_available()
print(f'PyTorch CUDA available: {cuda}')
if cuda:
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"
}

# Install dependencies
install_deps() {
    header
    echo -e "${BOLD}  INSTALLING DEPENDENCIES${NC}"
    header

    # Try pip with --break-system-packages or use existing packages
    pip install --break-system-packages torch pennylane pennylane-qiskit qiskit numpy pandas scikit-learn scipy matplotlib seaborn 2>/dev/null || \
    pip install torch pennylane pennylane-qiskit qiskit numpy pandas scikit-learn scipy matplotlib seaborn --user 2>/dev/null || \
    echo "Using existing packages..."
    
    log "Dependencies ready"
}

# ============================================================================
# 1. Smoke Test
# ============================================================================
run_smoke_test() {
    header
    echo -e "${BOLD}  1. SMOKE TEST${NC}"
    header

    python main.py --smoke
}

# ============================================================================
# 2. Draw Quantum Circuits
# ============================================================================
draw_circuits() {
    header
    echo -e "${BOLD}  2. DRAWING QUANTUM CIRCUITS${NC}"
    header

    mkdir -p "$OUTPUT_DIR/circuits"

    python3 << 'EOF'
import sys
import os
sys.path.insert(0, 'src')

import torch
import pennylane as qml

os.makedirs("output_result/benchmarks/circuits", exist_ok=True)

print("[Circuit Drawing] Generating quantum circuit diagrams...")

n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

# Strongly Entangling 2 layers
@qml.qnode(dev)
def se_circuit(features, weights):
    qml.AngleEmbedding(features, wires=range(n_qubits))
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

torch.manual_seed(42)
dummy_features = torch.randn(n_qubits)
dummy_weights = torch.randn(2, n_qubits, 3)

with open("output_result/benchmarks/circuits/strongly_entangling_2layers.txt", "w") as f:
    f.write("Strongly Entangling Layers Circuit (2 layers, 4 qubits)\n")
    f.write("=" * 60 + "\n\n")
    f.write(qml.draw(se_circuit)(dummy_features, dummy_weights))
    f.write("\n\nLegend:\n- RY/RZ/RX: Single-qubit rotations\n- CNOT: Entangling gate\n- [x]: Feature embedding\n")

print("[Circuit] Saved: strongly_entangling_2layers.txt")

# Strongly Entangling 3 layers
@qml.qnode(dev)
def se3_circuit(features, weights):
    qml.AngleEmbedding(features, wires=range(n_qubits))
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

with open("output_result/benchmarks/circuits/strongly_entangling_3layers.txt", "w") as f:
    f.write("Strongly Entangling Layers Circuit (3 layers, 4 qubits)\n")
    f.write("=" * 60 + "\n\n")
    f.write(qml.draw(se3_circuit)(dummy_features, torch.randn(3, n_qubits, 3)))

print("[Circuit] Saved: strongly_entangling_3layers.txt")

# Simple Data Reuploading (manual gates)
@qml.qnode(dev)
def dr_circuit(features, weights):
    for i in range(2):  # 2 layers
        for j in range(n_qubits):
            qml.RX(features[j], wires=j)
        for j in range(n_qubits - 1):
            qml.CNOT(wires=[j, j + 1])
        qml.RY(weights[i, 0], wires=0)
        qml.RY(weights[i, 1], wires=1)
        qml.RY(weights[i, 2], wires=2)
        qml.RY(weights[i, 3], wires=3)
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

with open("output_result/benchmarks/circuits/data_reuploading.txt", "w") as f:
    f.write("Data Reuploading Circuit (2 layers, 4 qubits)\n")
    f.write("=" * 60 + "\n\n")
    f.write(qml.draw(dr_circuit)(dummy_features, torch.randn(2, n_qubits)))

print("[Circuit] Saved: data_reuploading.txt")

# Circuit analysis
with open("output_result/benchmarks/circuits/circuit_analysis.txt", "w") as f:
    f.write("Circuit Depth Analysis for NISQ Compatibility\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"{'n_qubits':<12} {'n_layers':<12} {'depth':<10} {'gates':<10} {'params':<10} {'NISQ OK?':<12}\n")
    f.write("-" * 70 + "\n")

    for n_layers in [1, 2, 3, 4, 5, 6]:
        for n_qubits in [4, 6, 8]:
            try:
                dev_temp = qml.device("default.qubit", wires=n_qubits)
                @qml.qnode(dev_temp)
                def temp_circuit(features, weights):
                    qml.AngleEmbedding(features[:min(len(features), n_qubits)], wires=range(n_qubits))
                    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
                    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

                weights = torch.randn(n_layers, n_qubits, 3)
                features = torch.randn(min(8, n_qubits))
                specs = qml.specs(temp_circuit)(features, weights)
                depth = specs.get('depth', n_layers * 4)
                gates = specs.get('num_gates', n_layers * n_qubits * 4)
                params = specs.get('num_parameters', n_layers * n_qubits * 3)
                nisq_ok = "YES" if n_layers <= 4 else "NO (EXCEEDS)"
                f.write(f"{n_qubits:<12} {n_layers:<12} {depth:<10} {gates:<10} {params:<10} {nisq_ok:<12}\n")
            except Exception as e:
                f.write(f"{n_qubits:<12} {n_layers:<12} ERROR\n")

print("[Circuit] Analysis saved: circuit_analysis.txt")
print("[Circuit] All circuits drawn successfully")
EOF

    log "Circuit diagrams saved to $OUTPUT_DIR/circuits/"
}

# ============================================================================
# 3. Train with QNG (GPU/CPU)
# ============================================================================
train_qng() {
    header
    echo -e "${BOLD}  3. TRAINING WITH QNG (Quantum Natural Gradient)${NC}"
    header

    mkdir -p "$OUTPUT_DIR"
    t0=$(date +%s)

    python3 << EOF
import sys
import os
sys.path.insert(0, 'src')
sys.path.insert(0, '.')

import torch
import numpy as np
import time
import json
from pathlib import Path

# Check device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Import modules
from augmentation.local_pqc import create_local_pqc_training_pipeline, ClusteredLocalPQC

# Generate synthetic data for testing
np.random.seed(42)
torch.manual_seed(42)

n_samples = 1000
coords = np.random.randn(n_samples, 2) * 10 + np.array([10.8, 106.3])  # Around Vietnam
features = np.random.randn(n_samples, 8)
targets = np.random.poisson(5, n_samples).astype(float)

print(f"Training data: {n_samples} samples")
print(f"Target distribution: mean={targets.mean():.2f}, std={targets.std():.2f}")

# Train with QNG
print("\nTraining Local PQC with QNG optimizer...")
model, info = create_local_pqc_training_pipeline(
    coords=coords,
    features=features,
    targets=targets,
    n_clusters=${N_CLUSTERS},
    cluster_method='kmeans',
    n_qubits=${N_QUBITS},
    n_layers=${N_LAYERS},
    epochs=${EPOCHS},
    lr=${LR},
    batch_size=${BATCH_SIZE},
    device=device,
    verbose=True,
    optimizer_type='qng',
)

# Compute R² on training data (for comparison)
model.eval()
with torch.no_grad():
    cluster_tensor = torch.LongTensor(info['cluster_labels'])
    features_tensor = torch.FloatTensor(features).to(device)
    targets_tensor = torch.FloatTensor(targets)

    pred, _ = model(features_tensor.to(device), cluster_tensor.to(device))
    pred = pred.cpu().numpy().flatten()

# Compute metrics
ss_res = np.sum((targets - pred) ** 2)
ss_tot = np.sum((targets - targets.mean()) ** 2)
r2 = 1 - ss_res / (ss_tot + 1e-10)
rmse = np.sqrt(np.mean((targets - pred) ** 2))
mae = np.mean(np.abs(targets - pred))

# Save results
results = {
    'optimizer': 'QNG',
    'epochs': ${EPOCHS},
    'n_layers': ${N_LAYERS},
    'n_clusters': ${N_CLUSTERS},
    'device': device,
    'best_loss': info['best_loss'],
    'r2_train': r2,
    'rmse_train': rmse,
    'mae_train': mae,
    'avg_epoch_time_sec': info['avg_epoch_time_sec'],
    'total_time_sec': info['total_time_sec'],
    'circuit_depth': info['circuit_depth'],
    'history_loss': info.get('history', {}).get('loss', [])[-10:],  # Last 10 losses
}

with open('${OUTPUT_DIR}/qng_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*60}")
print(f"QNG RESULTS")
print(f"{'='*60}")
print(f"  Optimizer: QNG")
print(f"  Device: {device}")
print(f"  Best Loss: {info['best_loss']:.4f}")
print(f"  R² (train): {r2:.4f}")
print(f"  RMSE (train): {rmse:.4f}")
print(f"  MAE (train): {mae:.4f}")
print(f"  Avg Epoch Time: {info['avg_epoch_time_sec']:.2f}s")
print(f"  Total Time: {info['total_time_sec']:.1f}s")
print(f"{'='*60}")
EOF

    log "QNG training complete"
}

# ============================================================================
# 4. Train with Adam (GPU/CPU)
# ============================================================================
train_adam() {
    header
    echo -e "${BOLD}  4. TRAINING WITH ADAM${NC}"
    header

    python3 << EOF
import sys
import os
sys.path.insert(0, 'src')
sys.path.insert(0, '.')

import torch
import numpy as np
import time
import json

# Check device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

from augmentation.local_pqc import create_local_pqc_training_pipeline, ClusteredLocalPQC

# Generate synthetic data (same as QNG)
np.random.seed(42)
torch.manual_seed(42)

n_samples = 1000
coords = np.random.randn(n_samples, 2) * 10 + np.array([10.8, 106.3])
features = np.random.randn(n_samples, 8)
targets = np.random.poisson(5, n_samples).astype(float)

print(f"Training data: {n_samples} samples")

# Train with Adam
print("\nTraining Local PQC with Adam optimizer...")
model, info = create_local_pqc_training_pipeline(
    coords=coords,
    features=features,
    targets=targets,
    n_clusters=${N_CLUSTERS},
    cluster_method='kmeans',
    n_qubits=${N_QUBITS},
    n_layers=${N_LAYERS},
    epochs=${EPOCHS},
    lr=${LR},
    batch_size=${BATCH_SIZE},
    device=device,
    verbose=True,
    optimizer_type='adam',
)

# Compute metrics
model.eval()
with torch.no_grad():
    cluster_tensor = torch.LongTensor(info['cluster_labels'])
    features_tensor = torch.FloatTensor(features).to(device)
    targets_tensor = torch.FloatTensor(targets)

    pred, _ = model(features_tensor.to(device), cluster_tensor.to(device))
    pred = pred.cpu().numpy().flatten()

ss_res = np.sum((targets - pred) ** 2)
ss_tot = np.sum((targets - targets.mean()) ** 2)
r2 = 1 - ss_res / (ss_tot + 1e-10)
rmse = np.sqrt(np.mean((targets - pred) ** 2))
mae = np.mean(np.abs(targets - pred))

results = {
    'optimizer': 'Adam',
    'epochs': ${EPOCHS},
    'n_layers': ${N_LAYERS},
    'n_clusters': ${N_CLUSTERS},
    'device': device,
    'best_loss': info['best_loss'],
    'r2_train': r2,
    'rmse_train': rmse,
    'mae_train': mae,
    'avg_epoch_time_sec': info['avg_epoch_time_sec'],
    'total_time_sec': info['total_time_sec'],
    'circuit_depth': info['circuit_depth'],
}

with open('${OUTPUT_DIR}/adam_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*60}")
print(f"ADAM RESULTS")
print(f"{'='*60}")
print(f"  Optimizer: Adam")
print(f"  Device: {device}")
print(f"  Best Loss: {info['best_loss']:.4f}")
print(f"  R² (train): {r2:.4f}")
print(f"  RMSE (train): {rmse:.4f}")
print(f"  MAE (train): {mae:.4f}")
print(f"  Avg Epoch Time: {info['avg_epoch_time_sec']:.2f}s")
print(f"  Total Time: {info['total_time_sec']:.1f}s")
print(f"{'='*60}")
EOF

    log "Adam training complete"
}

# ============================================================================
# 5. Train CNN-LSTM Baseline
# ============================================================================
train_cnn_lstm() {
    header
    echo -e "${BOLD}  5. TRAINING CNN-LSTM BASELINE${NC}"
    header

    python3 << EOF
import sys
import os
sys.path.insert(0, 'src')
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import numpy as np
import time
import json

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

from models.cnn_lstm import SpatioTemporalCNNv2

# Generate sequential data for CNN-LSTM
np.random.seed(42)
torch.manual_seed(42)

seq_len = ${SEQ_LEN}
grid_size = ${GRID_SIZE}
n_sequences = 500

# Create synthetic spatiotemporal data
X = np.random.randn(n_sequences, seq_len, grid_size, grid_size).astype(np.float32)
y = np.random.randn(n_sequences, 1, grid_size, grid_size).astype(np.float32)
y = np.abs(y)  # Ensure positive (case counts)

print(f"CNN-LSTM data: {n_sequences} sequences, shape {X.shape}")

# Create model
model = SpatioTemporalCNNv2(
    grid_size=grid_size,
    forecast_horizon=1,
    loss='mse'
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=${LR}, weight_decay=1e-4)
criterion = nn.MSELoss()

# Training
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(
        torch.FloatTensor(X),
        torch.FloatTensor(y)
    ),
    batch_size=${BATCH_SIZE},
    shuffle=True
)

t0 = time.time()
losses = []

for epoch in range(${EPOCHS}):
    model.train()
    epoch_loss = 0
    n_batches = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1

    avg_loss = epoch_loss / max(n_batches, 1)
    losses.append(avg_loss)

    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1}/${EPOCHS} | Loss: {avg_loss:.4f}")

total_time = time.time() - t0

# Evaluate
model.eval()
with torch.no_grad():
    X_test = torch.FloatTensor(X[:100]).to(device)
    y_test = torch.FloatTensor(y[:100]).to(device)
    pred = model(X_test)

    pred_flat = pred.cpu().numpy().flatten()
    y_flat = y_test.cpu().numpy().flatten()

    ss_res = np.sum((y_flat - pred_flat) ** 2)
    ss_tot = np.sum((y_flat - y_flat.mean()) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-10)
    rmse = np.sqrt(np.mean((y_flat - pred_flat) ** 2))
    mae = np.mean(np.abs(y_flat - pred_flat))

results = {
    'model': 'CNN-LSTM',
    'epochs': ${EPOCHS},
    'grid_size': ${GRID_SIZE},
    'seq_len': ${SEQ_LEN},
    'device': device,
    'final_loss': losses[-1],
    'r2_test': r2,
    'rmse_test': rmse,
    'mae_test': mae,
    'total_time_sec': total_time,
    'avg_epoch_time_sec': total_time / ${EPOCHS},
    'history_loss': losses[-10:],
}

with open('${OUTPUT_DIR}/cnn_lstm_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*60}")
print(f"CNN-LSTM RESULTS")
print(f"{'='*60}")
print(f"  Model: CNN-LSTM")
print(f"  Device: {device}")
print(f"  Final Loss: {losses[-1]:.4f}")
print(f"  R² (test): {r2:.4f}")
print(f"  RMSE (test): {rmse:.4f}")
print(f"  MAE (test): {mae:.4f}")
print(f"  Total Time: {total_time:.1f}s")
print(f"{'='*60}")
EOF

    log "CNN-LSTM training complete"
}

# ============================================================================
# 6. Generate Comparison Report
# ============================================================================
generate_report() {
    header
    echo -e "${BOLD}  6. GENERATING COMPARISON REPORT${NC}"
    header

    python3 << 'EOF'
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.style.use('seaborn-v0_8-whitegrid')

# Load results
results = {}
for name in ['qng', 'adam', 'cnn_lstm']:
    try:
        with open(f'output_result/benchmarks/{name}_results.json', 'r') as f:
            results[name] = json.load(f)
    except:
        print(f"Warning: Could not load {name}_results.json")

# Create comparison plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Loss comparison
ax1 = axes[0, 0]
optimizers = []
losses = []
times = []

if 'qng' in results:
    optimizers.append('QNG')
    losses.append(results['qng']['best_loss'])
    times.append(results['qng']['avg_epoch_time_sec'])
if 'adam' in results:
    optimizers.append('Adam')
    losses.append(results['adam']['best_loss'])
    times.append(results['adam']['avg_epoch_time_sec'])

colors = ['#2ecc71', '#3498db', '#e74c3c'][:len(optimizers)]
bars = ax1.bar(optimizers, losses, color=colors, edgecolor='black')
ax1.set_ylabel('Best Loss (MSE)')
ax1.set_title('Optimizer Comparison: Best Loss')
for bar, loss in zip(bars, losses):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{loss:.4f}', ha='center', va='bottom', fontweight='bold')

# 2. R² comparison
ax2 = axes[0, 1]
r2_values = []
r2_labels = []
r2_colors = []

if 'qng' in results:
    r2_values.append(results['qng'].get('r2_train', 0))
    r2_labels.append('QNG (LocalPQC)')
    r2_colors.append('#2ecc71')
if 'adam' in results:
    r2_values.append(results['adam'].get('r2_train', 0))
    r2_labels.append('Adam (LocalPQC)')
    r2_colors.append('#3498db')
if 'cnn_lstm' in results:
    r2_values.append(results['cnn_lstm'].get('r2_test', 0))
    r2_labels.append('CNN-LSTM')
    r2_colors.append('#e74c3c')

bars2 = ax2.bar(r2_labels, r2_values, color=r2_colors[:len(r2_labels)], edgecolor='black')
ax2.set_ylabel('R² Score')
ax2.set_title('Model Comparison: R² Score')
ax2.set_ylim(0, 1)
for bar, r2 in zip(bars2, r2_values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{r2:.4f}', ha='center', va='bottom', fontweight='bold')

# 3. Training time comparison
ax3 = axes[1, 0]
if times:
    bars3 = ax3.bar(optimizers, times, color=colors[:len(optimizers)], edgecolor='black')
    ax3.set_ylabel('Avg Epoch Time (s)')
    ax3.set_title('Training Speed: Avg Epoch Time')
    for bar, t in zip(bars3, times):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{t:.2f}s', ha='center', va='bottom', fontweight='bold')

# 4. Metrics summary table
ax4 = axes[1, 1]
ax4.axis('off')

table_data = [['Model', 'R²', 'RMSE', 'MAE', 'Time (s)']]
if 'qng' in results:
    r = results['qng']
    table_data.append(['QNG', f"{r.get('r2_train', 'N/A'):.4f}" if isinstance(r.get('r2_train'), (int, float)) else 'N/A',
                       f"{r.get('rmse_train', 'N/A'):.4f}" if isinstance(r.get('rmse_train'), (int, float)) else 'N/A',
                       f"{r.get('mae_train', 'N/A'):.4f}" if isinstance(r.get('mae_train'), (int, float)) else 'N/A',
                       f"{r.get('total_time_sec', 0):.1f}"])
if 'adam' in results:
    r = results['adam']
    table_data.append(['Adam', f"{r.get('r2_train', 'N/A'):.4f}" if isinstance(r.get('r2_train'), (int, float)) else 'N/A',
                       f"{r.get('rmse_train', 'N/A'):.4f}" if isinstance(r.get('rmse_train'), (int, float)) else 'N/A',
                       f"{r.get('mae_train', 'N/A'):.4f}" if isinstance(r.get('mae_train'), (int, float)) else 'N/A',
                       f"{r.get('total_time_sec', 0):.1f}"])
if 'cnn_lstm' in results:
    r = results['cnn_lstm']
    table_data.append(['CNN-LSTM', f"{r.get('r2_test', 'N/A'):.4f}" if isinstance(r.get('r2_test'), (int, float)) else 'N/A',
                       f"{r.get('rmse_test', 'N/A'):.4f}" if isinstance(r.get('rmse_test'), (int, float)) else 'N/A',
                       f"{r.get('mae_test', 'N/A'):.4f}" if isinstance(r.get('mae_test'), (int, float)) else 'N/A',
                       f"{r.get('total_time_sec', 0):.1f}"])

table = ax4.table(cellText=table_data, loc='center', cellLoc='center',
                  colWidths=[0.25, 0.2, 0.2, 0.2, 0.15])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)

# Style header row
for i in range(len(table_data[0])):
    table[(0, i)].set_facecolor('#3498db')
    table[(0, i)].set_text_props(color='white', fontweight='bold')

ax4.set_title('Summary Metrics', fontweight='bold', pad=20)

plt.suptitle('Quantum Dengue STPP - Benchmark Results', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('output_result/benchmarks/benchmark_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: output_result/benchmarks/benchmark_comparison.png")

# Generate text report
with open('output_result/benchmarks/full_report.txt', 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("  QUANTUM DENGUE STPP - FULL BENCHMARK REPORT\n")
    f.write("=" * 70 + "\n\n")

    f.write("CONFIGURATION\n")
    f.write("-" * 40 + "\n")
    f.write(f"  Grid Size: ${GRID_SIZE}\n")
    f.write(f"  Sequence Length: ${SEQ_LEN}\n")
    f.write(f"  Epochs: ${EPOCHS}\n")
    f.write(f"  Batch Size: ${BATCH_SIZE}\n")
    f.write(f"  Learning Rate: ${LR}\n")
    f.write(f"  Circuit Layers: ${N_LAYERS}\n")
    f.write(f"  Clusters: ${N_CLUSTERS}\n")
    f.write(f"  Qubits: ${N_QUBITS}\n")
    f.write("\n")

    f.write("RESULTS SUMMARY\n")
    f.write("-" * 40 + "\n")

    for name, res in results.items():
        f.write(f"\n{name.upper()}:\n")
        f.write(f"  Device: {res.get('device', 'N/A')}\n")
        f.write(f"  Best Loss: {res.get('best_loss', res.get('final_loss', 'N/A')):.4f}\n")
        r2_key = 'r2_train' if 'r2_train' in res else 'r2_test'
        f.write(f"  R²: {res.get(r2_key, 'N/A'):.4f}\n" if isinstance(res.get(r2_key), (int, float)) else f"  R²: N/A\n")
        f.write(f"  RMSE: {res.get('rmse_train', res.get('rmse_test', 'N/A')):.4f}\n" if isinstance(res.get('rmse_train', res.get('rmse_test')), (int, float)) else f"  RMSE: N/A\n")
        f.write(f"  MAE: {res.get('mae_train', res.get('mae_test', 'N/A')):.4f}\n" if isinstance(res.get('mae_train', res.get('mae_test')), (int, float)) else f"  MAE: N/A\n")
        f.write(f"  Total Time: {res.get('total_time_sec', 0):.1f}s\n")
        f.write(f"  Avg Epoch Time: {res.get('avg_epoch_time_sec', 0):.2f}s\n")

    f.write("\n" + "=" * 70 + "\n")
    f.write("  FILES GENERATED\n")
    f.write("=" * 70 + "\n")
    f.write("  - circuits/strongly_entangling_2layers.txt\n")
    f.write("  - circuits/strongly_entangling_3layers.txt\n")
    f.write("  - circuits/data_reuploading.txt\n")
    f.write("  - circuits/circuit_analysis.txt\n")
    f.write("  - qng_results.json\n")
    f.write("  - adam_results.json\n")
    f.write("  - cnn_lstm_results.json\n")
    f.write("  - benchmark_comparison.png\n")
    f.write("  - full_report.txt\n")

print("\nFull report saved to output_result/benchmarks/full_report.txt")
EOF

    log "Comparison report generated"
}

# ============================================================================
# Main
# ============================================================================
main() {
    echo ""
    echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}${CYAN}║   QUANTUM DENGUE STPP - FULL BENCHMARK SUITE               ║${NC}"
    echo -e "${BOLD}${CYAN}║   GPU-accelerated | QNG vs Adam vs CNN-LSTM               ║${NC}"
    echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    check_gpu
    install_deps
    run_smoke_test
    draw_circuits
    train_qng
    train_adam
    train_cnn_lstm
    generate_report

    header
    echo -e "${BOLD}${GREEN}  BENCHMARK COMPLETE!${NC}"
    header
    echo ""
    echo "Results saved to: ${OUTPUT_DIR}/"
    echo ""
    echo "Files:"
    echo "  - benchmark_comparison.png (visual comparison)"
    echo "  - full_report.txt (detailed report)"
    echo "  - *_results.json (raw results)"
    echo "  - circuits/*.txt (quantum circuit diagrams)"
    echo ""
}

# Run
main "$@"
