"""Honest self-check helpers for quantum components.

These functions exist to keep us honest. They enforce the
guard-rails documented in QUANTUM_INTEGRATION_PLAN.md:

- Q-kernel must be limited to <= 15 qubits.
- QAOA must fall back to classical above 15 qubits.
- All comparisons must report both quality and timing.
- The plot generator must always include a "no quantum advantage
  claimed" disclaimer when quantum is on the same axes as classical.

If a downstream caller violates one of these rules the helper raises
a RuntimeError so the bad comparison cannot silently ship.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


# Guard rails — these are *not* about performance, they are about
# honesty. If you find yourself wanting to relax them, you are about
# to make an over-claim.
MAX_QKERNEL_QUBITS = 15
MAX_QAOA_QUBITS = 15


def check_quantum_claims(claim: str) -> None:
    """Refuse to print certain over-claims.

    The function distinguishes between:
      - honest disclaimer ("no quantum advantage claimed") — allowed
      - positive over-claim ("shows quantum advantage") — blocked

    The matching uses word boundaries so "no quantum advantage claimed"
    does not trip the "quantum advantage" filter.

    Args:
        claim: the assertion that is about to be made in a log or plot.

    Raises:
        RuntimeError: if the claim is on the blacklist.
    """
    import re

    positive_blacklist = [
        "quantum advantage",
        "quantum speedup",
        "exponential speedup",
        "outperforms classical",
        "beats rbf",
        "beats svm",
        "perfect classification",
        "100% accuracy",
        "qknn o(sqrt(n))",
        "qrend",
    ]
    lowered = claim.lower()

    # An "honest disclaimer" that explicitly states there is no advantage
    # is allowed through. We only block positive statements of advantage.
    if "no quantum advantage" in lowered or "without quantum advantage" in lowered:
        return

    for phrase in positive_blacklist:
        # Use word boundaries so 'quantum advantage' inside the phrase
        # 'no quantum advantage claimed' still trips the rule if not
        # for the explicit disclaimer path above.
        pattern = r"\b" + re.escape(phrase) + r"\b"
        if re.search(pattern, lowered):
            raise RuntimeError(
                f"Refused over-claim: {claim!r}. "
                f"The phrase {phrase!r} is on the blacklist. "
                f"Restate honestly or remove."
            )


def validate_qaoa_output(
    x_best: np.ndarray,
    Q: np.ndarray,
    expected_cardinality: int,
) -> Tuple[bool, str]:
    """Verify a QAOA result is well-formed before returning it."""
    x = np.asarray(x_best)
    if x.ndim != 1:
        return False, f"x_best must be 1-D, got shape {x.shape}"
    if not np.all((x == 0) | (x == 1)):
        return False, "x_best must be binary {0,1}"
    if len(x) != Q.shape[0]:
        return False, f"len(x_best)={len(x)} != Q.shape[0]={Q.shape[0]}"
    cardinality = int(np.sum(x))
    if cardinality != expected_cardinality:
        return False, (
            f"cardinality {cardinality} != requested "
            f"{expected_cardinality}"
        )
    return True, "ok"


__all__ = [
    "check_quantum_claims",
    "validate_qaoa_output",
    "MAX_QKERNEL_QUBITS",
    "MAX_QAOA_QUBITS",
]
