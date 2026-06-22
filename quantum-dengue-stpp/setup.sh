#!/bin/bash
# =============================================================================
# Setup script for Quantum Dengue STPP
# =============================================================================

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Quantum Dengue STPP - Setup Environment${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check Python version
echo -e "${YELLOW}[1/6] Checking Python version...${NC}"
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ $(echo "$PYTHON_VERSION >= 3.10" | bc -l 2>/dev/null || python3 -c "print(float('$PYTHON_VERSION') >= 3.10)") == "True" ]]; then
    echo -e "${GREEN}✓${NC} Python $PYTHON_VERSION"
else
    echo -e "${RED}✗${NC} Python $PYTHON_VERSION found, but >= 3.10 required"
    exit 1
fi

# Create virtual environment
echo -e "${YELLOW}[2/6] Creating virtual environment...${NC}"
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo -e "${GREEN}✓${NC} Created .venv"
else
    echo -e "${GREEN}✓${NC} Using existing .venv"
fi

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
echo -e "${YELLOW}[3/6] Upgrading pip...${NC}"
pip install --upgrade pip > /dev/null 2>&1
echo -e "${GREEN}✓${NC} pip upgraded"

# Install main requirements
echo -e "${YELLOW}[4/6] Installing requirements...${NC}"
pip install -r requirements.txt > /dev/null 2>&1
echo -e "${GREEN}✓${NC} Main requirements installed"

# Install QuApp (optional)
echo -e "${YELLOW}[5/6] Installing QuApp CLI...${NC}"
pip install quapp>=0.1.5 > /dev/null 2>&1 && echo -e "${GREEN}✓${NC} QuApp CLI installed" || echo -e "${YELLOW}⚠${NC} QuApp CLI skipped (optional)"

# Install dev requirements if exists
if [ -f "requirements-dev.txt" ]; then
    pip install -r requirements-dev.txt > /dev/null 2>&1 && echo -e "${GREEN}✓${NC} Dev requirements installed"
fi

# Verify installation
echo -e "${YELLOW}[6/6] Verifying installation...${NC}"
python -c "
import sys
import importlib.util

modules = ['torch', 'pennylane', 'numpy', 'pandas', 'scipy']
all_ok = True
for mod in modules:
    spec = importlib.util.find_spec(mod)
    if spec:
        m = importlib.import_module(mod)
        print(f'  ✓ {mod}: {m.__version__}')
    else:
        print(f'  ✗ {mod}: not found')
        all_ok = False

if all_ok:
    print('All core modules verified!')
else:
    print('Some modules missing. Run: pip install -r requirements.txt')
"

echo
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Setup Complete!${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo
echo -e "To activate the environment, run:"
echo -e "  ${GREEN}source .venv/bin/activate${NC}"
echo
echo -e "To test quantum circuits:"
echo -e "  ${GREEN}cd quapp && python -c \"from handler import handler; print(handler({'circuit_type': 'qbm', 'n_qubits': 4, 'n_layers': 2, 'shots': 100}))\"${NC}"
echo
echo -e "To run tests:"
echo -e "  ${GREEN}python -m pytest tests/ -v${NC}"
echo
