#!/bin/bash
# Quantum Dengue STPP - Quick Run Script
# Usage: ./run.sh [mode] [options]
#
# Modes:
#   smoke     - Run smoke test (default)
#   train     - Train Local PQC with QNG
#   train-adam - Train Local PQC with Adam
#   full      - Full pipeline (requires data directory)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check Python environment
check_env() {
    print_status "Checking environment..."

    if ! command -v python &> /dev/null; then
        print_error "Python not found. Please install Python 3.9+"
        exit 1
    fi

    python_version=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    print_status "Python version: $python_version"

    # Check key packages
    python -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null || print_warning "PyTorch not found"
    python -c "import pennylane; print(f'PennyLane: {pennylane.__version__}')" 2>/dev/null || print_warning "PennyLane not found"
    python -c "import numpy; print(f'NumPy: {numpy.__version__}')" 2>/dev/null || print_warning "NumPy not found"
}

# Install dependencies if needed
install_deps() {
    print_status "Installing dependencies..."
    pip install torch pennylane pennylane-qiskit qiskit numpy pandas scikit-learn scipy --quiet
}

# Run smoke test
run_smoke() {
    print_status "Running smoke test..."
    python main.py --smoke
}

# Train Local PQC with QNG
run_train_qng() {
    print_status "Training Local PQC with QNG optimizer..."
    python main.py --train-local-pqc --optimizer qng --epochs 50 --n-layers 3 --n-clusters 8
}

# Train Local PQC with Adam
run_train_adam() {
    print_status "Training Local PQC with Adam optimizer..."
    python main.py --train-local-pqc --optimizer adam --epochs 50 --n-layers 3 --n-clusters 8
}

# Compare QNG vs Adam
run_compare() {
    print_status "Running QNG vs Adam comparison..."

    echo ""
    echo "=========================================="
    echo "  BENCHMARK: QNG vs Adam"
    echo "=========================================="
    echo ""

    echo ">>> Training with QNG..."
    run_train_qng

    echo ""
    echo ">>> Training with Adam..."
    run_train_adam

    echo ""
    echo "=========================================="
    echo "  COMPARISON COMPLETE"
    echo "=========================================="
    echo ""
    echo "Compare the output above to evaluate:"
    echo "  - Convergence speed (epochs to best loss)"
    echo "  - Final best loss"
    echo "  - Average epoch time"
}

# Show help
show_help() {
    echo "Quantum Dengue STPP - Quick Run Script"
    echo ""
    echo "Usage: $0 [mode] [options]"
    echo ""
    echo "Modes:"
    echo "  smoke        Run smoke test (default if no mode specified)"
    echo "  train        Train Local PQC with QNG optimizer"
    echo "  train-adam   Train Local PQC with Adam optimizer"
    echo "  compare      Run both QNG and Adam, compare results"
    echo "  help         Show this help message"
    echo ""
    echo "Options:"
    echo "  --epochs N       Number of training epochs (default: 50)"
    echo "  --n-layers N     Number of circuit layers (default: 3, max: 4)"
    echo "  --n-clusters N   Number of spatial clusters (default: 8)"
    echo ""
    echo "Examples:"
    echo "  $0 smoke"
    echo "  $0 train"
    echo "  $0 train --epochs 100 --n-layers 4"
    echo "  $0 compare"
}

# Parse arguments
MODE="${1:-smoke}"
shift 2>/dev/null || true

case "$MODE" in
    smoke)
        check_env
        run_smoke
        ;;
    train)
        check_env
        run_train_qng
        ;;
    train-adam)
        check_env
        run_train_adam
        ;;
    compare)
        check_env
        run_compare
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown mode: $MODE"
        echo ""
        show_help
        exit 1
        ;;
esac

echo ""
print_status "Done!"
