"""q_dengue_epidemiology - Epidemiological hotspot detection with quantum algorithms.

Modules:
- graph_dien_bien: Classical GIS preprocessing, 130 communes (synthetic)
- qpie_encoder: QPIE quantum state preparation
- durr_hoyer_max: Grover-based max finding
- lackadaisical_walk: Grover amplification for top-K hotspots
- qpia_epidemiology: Quantum Path Integral Approach (experimental)
- pipeline: Hybrid orchestrator

On real quantum hardware: O(√N) query complexity.
On classical simulator: classical numpy (no speedup, same algorithm).
"""
from .graph_dien_bien import build_synthetic_dien_bien, DienBienGraph, Commune
from .qpie_encoder import encode_risk_qpie, qpie_qnode
from .durr_hoyer_max import dur_hoyer_max_finding
from .lackadaisical_walk import multi_hotspot_detection
from .pipeline import run_full_pipeline
from .qpia_epidemiology import (
    qpia_search,
    qpia_grover_hybrid,
    qpia_index_case_finding,
    QPIAResult,
    PathState,
    enumerate_paths,
    compute_path_amplitudes,
    aggregate_to_nodes,
    compute_interference_analysis,
)

__all__ = [
    "build_synthetic_dien_bien",
    "DienBienGraph",
    "Commune",
    "encode_risk_qpie",
    "qpie_qnode",
    "dur_hoyer_max_finding",
    "multi_hotspot_detection",
    "run_full_pipeline",
    # QPIA exports
    "qpia_search",
    "qpia_grover_hybrid",
    "qpia_index_case_finding",
    "QPIAResult",
    "PathState",
    "enumerate_paths",
    "compute_path_amplitudes",
    "aggregate_to_nodes",
    "compute_interference_analysis",
]
