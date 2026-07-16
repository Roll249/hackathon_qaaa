#!/bin/bash
# Q-STPP v15 (corrected) — convenience runner.
#
# Runs the fair SOP comparison (classical MH / Grover-inspired / QAOA-inspired).
# There is no quantum backend; only numpy/scipy/matplotlib are required.
#
# Usage:
#   ./run.sh                 # defaults: seeds 1..5, N in {20,30,50}
#   ./run.sh smoke           # tiny/fast run for a sanity check
#   ./run.sh --seeds 1 2 3 --n_events 20 40 60 --evals_per_perm 400
#   ./run.sh help

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Writable Matplotlib config dir (avoids the "config dir not writable" warning
# on machines with a read-only/absent HOME). The Python script also sets this.
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/mplconfig}"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

check_env() {
    command -v python3 >/dev/null 2>&1 || { error "python3 not found"; exit 1; }
    python3 -c "import numpy, scipy, matplotlib" 2>/dev/null \
        || warn "missing deps — run: pip install -r requirements.txt"
}

case "${1:-run}" in
    help|--help|-h)
        sed -n '2,12p' "$0"
        ;;
    smoke)
        check_env
        info "Smoke test (2 seeds, N=20, small budget)..."
        python3 run_q_stpp_v15_fair.py --seeds 1 2 --n_events 20 --evals_per_perm 40
        ;;
    run)
        check_env
        info "Fair SOP comparison (defaults)..."
        python3 run_q_stpp_v15_fair.py
        ;;
    *)
        # pass all args straight through to the script
        check_env
        info "Fair SOP comparison (custom args)..."
        python3 run_q_stpp_v15_fair.py "$@"
        ;;
esac

info "Done."
