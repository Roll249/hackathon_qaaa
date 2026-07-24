#!/bin/bash
# =============================================================================
# Stress Test Script for QRC v2 - GPU Acceleration Testing
# =============================================================================
# This script runs QRC v2 with progressively larger configurations to test
# system limits and GPU acceleration capabilities.
#
# Usage:
#   ./stress_test.sh                    # Run all tests
#   ./stress_test.sh --quick           # Quick test (fewer configurations)
#   ./stress_test.sh --gpu-only         # GPU-only tests
#   ./stress_test.sh --small-only       # Small config tests only
#
# Requirements:
#   - pennylane >= 0.38.0
#   - pennylane-lightning.gpu (optional, for GPU acceleration)
#   - 16GB+ RAM recommended
#
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# Output directory
OUTPUT_DIR="$PROJECT_DIR/output_result/q_stpp_v18/stress_test"
mkdir -p "$OUTPUT_DIR"

# Log file
LOG_FILE="$OUTPUT_DIR/stress_test_$(date +%Y%m%d_%H%M%S).log"

# =============================================================================
# Configuration
# =============================================================================

# Stress test configurations (scaled for powerful machines)
# These configurations push the limits of quantum simulation

# Small configurations (baseline)
SMALL_CONFIGS=(
    "n_qubits=8,n_layers=3,n_internal=50"
    "n_qubits=8,n_layers=4,n_internal=50"
    "n_qubits=12,n_layers=3,n_internal=50"
)

# Medium configurations
MEDIUM_CONFIGS=(
    "n_qubits=12,n_layers=4,n_internal=50"
    "n_qubits=12,n_layers=5,n_internal=100"
    "n_qubits=16,n_layers=4,n_internal=50"
    "n_qubits=16,n_layers=5,n_internal=100"
)

# Large configurations (for powerful machines)
LARGE_CONFIGS=(
    "n_qubits=16,n_layers=6,n_internal=100"
    "n_qubits=20,n_layers=5,n_internal=100"
    "n_qubits=20,n_layers=6,n_internal=200"
    "n_qubits=24,n_layers=4,n_internal=100"
    "n_qubits=24,n_layers=5,n_internal=150"
)

# Extra large (for very powerful GPU machines)
XLARGE_CONFIGS=(
    "n_qubits=24,n_layers=6,n_internal=200"
    "n_qubits=28,n_layers=5,n_internal=150"
    "n_qubits=28,n_layers=6,n_internal=200"
    "n_qubits=32,n_layers=4,n_internal=200"
)

# Number of provinces to test in parallel
PARALLEL_PROVINCES=4

# Number of random seeds
N_SEEDS=3

# Test series length
N_WEEKS=300

# =============================================================================
# Helper Functions
# =============================================================================

log() {
    echo -e "${BLUE}[$(date +%H:%M:%S)]${NC} $1"
    echo "[$(date +%Y-%m-%d %H:%M:%S)] $1" >> "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[$(date +%H:%M:%S)] ✓${NC} $1"
    echo "[SUCCESS] $1" >> "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[$(date +%H:%M:%S)] ⚠${NC} $1"
    echo "[WARNING] $1" >> "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[$(date +%H:%M:%S)] ✗${NC} $1"
    echo "[ERROR] $1" >> "$LOG_FILE"
}

check_gpu() {
    if python3 -c "import pennylane as qml; dev = qml.device('lightning.gpu', wires=2); print('GPU available')" 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

check_dependencies() {
    log "Checking dependencies..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 not found"
        exit 1
    fi
    
    # Check PennyLane
    if ! python3 -c "import pennylane; print(f'PennyLane {pennylane.__version__}')" 2>/dev/null; then
        log_error "PennyLane not installed"
        exit 1
    fi
    
    # Check numpy
    if ! python3 -c "import numpy; print(f'NumPy {numpy.__version__}')" 2>/dev/null; then
        log_error "NumPy not installed"
        exit 1
    fi
    
    log_success "All dependencies available"
}

run_single_test() {
    local config="$1"
    local test_name="$2"
    local output_file="$OUTPUT_DIR/${test_name}_results.json"
    
    log "Running: $config"
    
    # Parse config
    n_qubits=$(echo "$config" | grep -oP '(?<=n_qubits=)\d+')
    n_layers=$(echo "$config" | grep -oP '(?<=n_layers=)\d+')
    n_internal=$(echo "$config" | grep -oP '(?<=n_internal=)\d+')
    
    # Calculate Hilbert space size
    hilbert_size=$((2 ** n_qubits))
    
    log "  Qubits: $n_qubits (Hilbert: ${hilbert_size}D)"
    log "  Layers: $n_layers"
    log "  Internal: $n_internal"
    
    # Run test
    start_time=$(date +%s)
    
    if python3 -c "
import sys
import time
import json
import numpy as np
sys.path.insert(0, 'src')

from quantum.quantum_reservoir_v2 import (
    ImprovedQRCConfig,
    direct_multihorizon_predict,
)

# Generate test data
np.random.seed(42)
n_weeks = $N_WEEKS
t = np.arange(n_weeks)

cases = 100 + 50*np.sin(2*np.pi*t/52) + 20*np.sin(4*np.pi*t/52) + 8*np.random.randn(n_weeks)
cases = np.maximum(cases, 0)

climate = {
    'temperature': 28 + 4*np.sin(2*np.pi*t/52) + 2*np.random.randn(n_weeks),
    'humidity': 75 + 10*np.sin(2*np.pi*t/52) + 5*np.random.randn(n_weeks),
    'rainfall': np.maximum(5 + 3*np.random.randn(n_weeks), 0),
}

config = ImprovedQRCConfig(
    n_qubits=$n_qubits,
    n_layers=$n_layers,
    n_internal=$n_internal,
    max_horizon=4,
    leaky=0.2,
    seed=42,
)

t_start = time.time()
result = direct_multihorizon_predict(
    cases=cases,
    climate_data=climate,
    config=config,
    train_fraction=0.7,
)
elapsed = time.time() - t_start

output = {
    'config': {
        'n_qubits': $n_qubits,
        'n_layers': $n_layers,
        'n_internal': $n_internal,
        'hilbert_space': $hilbert_size,
    },
    'results': {
        'mse_h1': result.mse_by_horizon[1],
        'mse_h2': result.mse_by_horizon[2],
        'mse_h4': result.mse_by_horizon[4],
        'nmse_h1': result.nmse_by_horizon[1],
        'train_time_s': result.train_time_s,
        'predict_time_s': result.predict_time_s,
        'total_time_s': elapsed,
        'n_params': result.n_params,
        'n_features': result.n_features,
    },
    'system_info': {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
}

with open('$output_file', 'w') as f:
    json.dump(output, f, indent=2)

print(f'DONE:{result.mse_by_horizon[1]:.4f}')
" 2>&1; then
        log_success "Completed: MSE=$(grep -oP '(?<=\"mse_h1\": )[0-9.]+' "$output_file")"
    else
        log_error "Test failed for: $config"
        return 1
    fi
}

run_stress_test() {
    local config_set="$1"
    local set_name="$2"
    
    log ""
    log "============================================================"
    log "Running $set_name Configuration Set"
    log "============================================================"
    
    local configs_var="${config_set}[@]"
    local configs=("${!configs_var}")
    
    local total=${#configs[@]}
    local current=0
    local failed=0
    
    for config in "${configs[@]}"; do
        current=$((current + 1))
        test_name="${set_name}_${current}"
        
        log ""
        log "--- [$current/$total] ---"
        
        if run_single_test "$config" "$test_name"; then
            : # Success
        else
            failed=$((failed + 1))
            log_warning "Test failed (total failures: $failed)"
        fi
    done
    
    log ""
    log "Completed $set_name: $((total - failed))/$total successful"
}

generate_summary() {
    log ""
    log "============================================================"
    log "Generating Stress Test Summary"
    log "============================================================"
    
    python3 << 'EOF'
import json
import os
import glob
import numpy as np

output_dir = os.environ.get('OUTPUT_DIR', 'output_result/q_stpp_v18/stress_test')
pattern = os.path.join(output_dir, '*_results.json')

results = []
for f in glob.glob(pattern):
    try:
        with open(f) as fp:
            results.append(json.load(fp))
    except:
        continue

if not results:
    print("No results found")
    exit()

# Aggregate by qubit count
by_qubits = {}
for r in results:
    nq = r['config']['n_qubits']
    if nq not in by_qubits:
        by_qubits[nq] = []
    by_qubits[nq].append(r['results'])

summary = {
    'total_tests': len(results),
    'by_qubit_count': {},
    'overall': {
        'mse_h1_mean': float(np.mean([r['results']['mse_h1'] for r in results])),
        'mse_h1_std': float(np.std([r['results']['mse_h1'] for r in results])),
        'train_time_mean': float(np.mean([r['results']['train_time_s'] for r in results])),
    }
}

for nq, res_list in sorted(by_qubits.items()):
    summary['by_qubit_count'][nq] = {
        'count': len(res_list),
        'mse_h1_mean': float(np.mean([r['mse_h1'] for r in res_list])),
        'mse_h1_std': float(np.std([r['mse_h1'] for r in res_list])),
        'train_time_mean': float(np.mean([r['train_time_s'] for r in res_list])),
    }

output_file = os.path.join(output_dir, 'stress_test_summary.json')
with open(output_file, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nSummary saved to: {output_file}")
print(f"\nOverall MSE (h=1): {summary['overall']['mse_h1_mean']:.4f} ± {summary['overall']['mse_h1_std']:.4f}")
print(f"Overall train time: {summary['overall']['train_time_mean']:.2f}s")
print(f"\nBy qubit count:")
for nq, stats in summary['by_qubit_count'].items():
    print(f"  {nq} qubits: {stats['count']} tests, MSE={stats['mse_h1_mean']:.4f}, time={stats['train_time_mean']:.2f}s")
EOF
}

# =============================================================================
# Parse Arguments
# =============================================================================

QUICK_MODE=false
GPU_ONLY=false
SMALL_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --gpu-only)
            GPU_ONLY=true
            shift
            ;;
        --small-only)
            SMALL_ONLY=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --quick       Quick test with fewer configurations"
            echo "  --gpu-only    Test GPU acceleration only"
            echo "  --small-only  Test small configurations only"
            echo "  --help, -h    Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Main
# =============================================================================

main() {
    log "============================================================"
    log "QRC v2 Stress Test"
    log "============================================================"
    log "Project: $PROJECT_DIR"
    log "Output: $OUTPUT_DIR"
    log "Log: $LOG_FILE"
    log "Started: $(date)"
    log "============================================================"
    
    # Check system
    check_dependencies
    
    # GPU check
    if check_gpu; then
        log_success "GPU (lightning.gpu) available"
        DEVICE="lightning.gpu"
    else
        log_warning "GPU not available, using default.qubit"
        DEVICE="default.qubit"
    fi
    
    # Export for Python scripts
    export OUTPUT_DIR
    
    # Run tests based on mode
    if [ "$QUICK_MODE" = true ]; then
        log "Running in QUICK mode"
        SMALL_CONFIGS=(
            "n_qubits=8,n_layers=3,n_internal=30"
            "n_qubits=8,n_layers=4,n_internal=50"
        )
        run_stress_test SMALL_CONFIGS "quick"
        
    elif [ "$SMALL_ONLY" = true ]; then
        run_stress_test SMALL_CONFIGS "small"
        
    elif [ "$GPU_ONLY" = true ]; then
        log "GPU stress tests would require pennylane-lightning.gpu"
        log "Installing: pip install pennylane-lightning[gpu]"
        # run_stress_test XLARGE_CONFIGS "gpu"
        
    else:
        # Full stress test
        run_stress_test SMALL_CONFIGS "small"
        run_stress_test MEDIUM_CONFIGS "medium"
        run_stress_test LARGE_CONFIGS "large"
        
        # Optional: XLARGE tests (uncomment for very powerful machines)
        # run_stress_test XLARGE_CONFIGS "xlarge"
    fi
    
    # Generate summary
    generate_summary
    
    log ""
    log "============================================================"
    log_success "Stress Test Complete!"
    log "Results saved to: $OUTPUT_DIR"
    log "Log file: $LOG_FILE"
    log "============================================================"
}

# Run main
main "$@"
