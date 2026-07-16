"""Quantum modules for RAPID-DENGUE v17.

Three honest, hackathon-feasible quantum components:
1. qkernel_hotspot          — Quantum kernel for STPP hotspot similarity
2. qubo_sop_selector        — QUBO-QAOA for SOP permutation subset selection
3. genuine_sop_quantum      — Factoradic amplitude amplification for SOP search

All run on PennyLane `default.qubit`. None claim wall-clock quantum
advantage over classical MH/Greedy at the sizes we operate on; the SOP
search component exposes the permutation-superposition structure and the
defensible sqrt(N!/M_tau) predicate-query scaling under a coherent
cost-oracle model.
"""

from .qkernel_hotspot import (
    QuantumKernelHotspot,
    quantum_kernel_matrix,
    quantum_kernel_circuit,
    quantum_knn_classify,
)
from .qubo_sop_selector import (
    QUBOSOPSelector,
    build_qubo_matrix,
    qubo_value,
    qaoa_solve,
)
from .genuine_sop_quantum import (
    SopQuantumResult,
    build_circuit,
    compute_L_summary,
    enqueue_all_costs,
    l_error,
    number_of_qubits,
    run_sop_quantum,
)
from .honest_assessment import (
    check_quantum_claims,
    validate_qaoa_output,
)

__all__ = [
    "QuantumKernelHotspot",
    "quantum_kernel_matrix",
    "quantum_kernel_circuit",
    "quantum_knn_classify",
    "QUBOSOPSelector",
    "build_qubo_matrix",
    "qubo_value",
    "qaoa_solve",
    "check_quantum_claims",
    "validate_qaoa_output",
    "SopQuantumResult",
    "build_circuit",
    "compute_L_summary",
    "enqueue_all_costs",
    "l_error",
    "number_of_qubits",
    "run_sop_quantum",
]
