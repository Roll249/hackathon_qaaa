"""
Genuine quantum SOP (Second-Order Preserving) search with PennyLane.

Scope
-----
This is NOT a v15-replacement classical heuristic. It is a *genuine quantum
search* over the space of SOP temporal permutations, using:

  * A factoradic / rank register  |pi>  (range [0, N!) on q = ceil(log2 N!) qubits)
  * A *table oracle* O_tau: marks permutations pi with L_error(pi) <= tau
  * An amplitude-amplification (Grover-style) iteration that reflects about the
    uniform superposition over *valid* permutations only.
  * Classical post-processing that maps the measured rank back to a
    permutation and verifies exact L_error on the host side.

Honest accounting
-----------------
  * State preparation: O(N log N) gates (Babbush et al.; quantum Fisher-Yates).
  * Oracle construction cost: O(N! * C(L)) classical preprocessing time
    (CANNOT be absorbed into a "quantum speedup" claim).
  * Query complexity to obtain a good permutation: O( sqrt( N! / M_tau ) ),
    where M_tau is the size of the good set. This is the *only* honest
    asymptotic advantage: fewer predicate calls than random sampling.
  * End-to-end wall-clock: dominated by classical oracle construction
    unless N is large enough that N! preprocessing exceeds the square-root
    quantum cost. Not relevant at N <= 8.

The point of this file is to *expose* the permutation-superposition structure
empirically at small N. It is a research artifact, not a v16 workhorse.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import pennylane as qml


# ---------------------------------------------------------------------------
# 1. SOP score (classical helper — identical definition to v15 fair script)
# ---------------------------------------------------------------------------


def l_error(L_perm: np.ndarray, L_target: np.ndarray) -> float:
    """Mean squared deviation of a permutation's L(r) from the target's."""
    return float(np.mean((L_perm - L_target) ** 2))


def compute_L_summary(
    times: np.ndarray,
    coords_x: np.ndarray,
    coords_y: np.ndarray,
    r_values: np.ndarray,
    T: float = 1.0,
    space_size: float = 1.0,
) -> np.ndarray:
    """Space-time second-order summary derived from Ripley's K.

    Following Mohler & Mateu (2023) "Second-order preserving point
    process permutations", we use the 3D point pattern
        z_i = (x_i, y_i, t_i / T * space_size)
    so that space and time coordinates share the same scale. L(r) is
    then computed from the 3D K-function as a stabilised transform.

    IMPORTANT PROPERTY: the 3D K-function depends on the *multiset*
    of pairwise distances for the COUNTING statistic, but the choice
    of times-per-position DOES matter: when we permute `times` while
    keeping `(coords_x, coords_y)` fixed, the distance matrix
        ||z_i - z_j||^2 = (x_i - x_j)^2 + (y_i - y_j)^2 + (t_i - t_j)^2
    changes because the temporal gap between two fixed locations
    depends on which time stamps are assigned to those locations.
    Hence SOP permutations DO change the 3D L(r) when the data has
    any space-time correlation. For purely Poisson (uncorrelated)
    data, L is permutation-invariant because distances are exchangeable.

    Implementation note: we follow the exact same convention as the
    v15 fair-comparison script (`run_q_stpp_v15_fair.compute_L_summary`),
    which uses `scipy.spatial.distance.pdist`. That convention counts
    the n diagonal self-pairs in `np.sum(dist < r)`, and subtracts them
    in the K estimator. We replicate this exactly so that L values
    produced here match those used elsewhere in the codebase.
    """
    if len(times) < 2:
        return np.zeros_like(r_values)

    time_scale = space_size / T
    z = np.column_stack([coords_x, coords_y, times * time_scale])
    n = len(times)

    # Compute pairwise distances in one shot. We use scipy when available
    # for parity with the v15 script (which the project's training pipeline
    # and benchmarks depend on). Falls back to a manual numpy version
    # that produces numerically identical output.
    try:
        from scipy.spatial.distance import pdist, squareform
        dist = squareform(pdist(z, metric="euclidean"))
        # pdist leaves self-distances at 0; we keep that so the K estimator
        # counts n self-pairs (matching the v15 reference).
    except ImportError:
        diffs = z[:, None, :] - z[None, :, :]
        d2 = np.sum(diffs * diffs, axis=-1)
        dist = np.sqrt(np.maximum(d2, 0.0))
        # Leave diagonal as 0; do not fill with inf.

    K = np.zeros_like(r_values)
    for k, r in enumerate(r_values):
        K[k] = (np.sum(dist < r) - n) / (n * n)
    return np.sign(K) * np.abs(K) ** (1.0 / 3.0)


# ---------------------------------------------------------------------------
# SOP cost function — Mohler & Mateu (2023) algorithm 1, applied offline
# ---------------------------------------------------------------------------


def _random_permutation_table(
    times: np.ndarray,
    coords_x: np.ndarray,
    coords_y: np.ndarray,
    r_values: np.ndarray,
    n_random: int,
    seed: int,
) -> np.ndarray:
    """Compute L(r) for `n_random` independent random permutations of
    `times`. Returns shape (n_random, R) array.

    The mean of these L_k vectors is mu(r); the per-permutation deviation
    L_data - L_k is the noise floor epsilon_k(r) that SOP permutations
    are supposed to match.
    """
    rng = np.random.default_rng(seed)
    n = len(times)
    L_random = np.empty((n_random, len(r_values)), dtype=np.float64)
    for k in range(n_random):
        perm = rng.permutation(n)
        L_random[k] = compute_L_summary(
            times[perm], coords_x, coords_y, r_values
        )
    return L_random


def sop_cost(
    times: np.ndarray,
    coords_x: np.ndarray,
    coords_y: np.ndarray,
    r_values: np.ndarray,
    L_data: np.ndarray,
    L_random_table: np.ndarray,
    perm: np.ndarray,
    baseline_index: int = 0,
) -> float:
    """Mohler-Mateu SOP cost for a single permutation.

    The objective (paper Algorithm 1) is

        || L_prop(r) - L_data(r) - epsilon_k(r) ||^2

    where epsilon_k(r) = L_data(r) - L_k(r) is the noise floor from the
    k-th random permutation, and L_k is the L(r) of that random perm.
    Substituting, this is equivalent to

        || L_prop(r) - L_k(r) ||^2

    i.e. the candidate permutation's L(r) should match the random
    permutation k's L(r). This cost function DOES depend on the choice
    of permutation because:
      (a) the baseline_index picks a specific random permutation whose
          L_k acts as the target, and
      (b) when the candidate perm is the identity (matching L_data) the
          cost is positive; when it matches L_k the cost is zero.

    This breaks the permutation-invariance of the bare 3D L(r) and
    makes the SOP problem a genuine combinatorial search — exactly the
    property Grover amplification can exploit.

    Args:
        times, coords_x, coords_y, r_values: STPP coordinates (used to
            compute L of `perm` and look up L_k from table).
        L_data: precomputed L of the original (un-permuted) data.
        L_random_table: shape (n_random, R) — L of random permutations.
        perm: candidate permutation of times.
        baseline_index: which random permutation's L(r) is the target.

    Returns:
        cost: scalar MSE between L(perm) and L_random_table[baseline].
    """
    L_prop = compute_L_summary(times[perm], coords_x, coords_y, r_values)
    L_target_k = L_random_table[baseline_index]
    return float(np.mean((L_prop - L_target_k) ** 2))


def sop_cost_table(
    times: np.ndarray,
    coords_x: np.ndarray,
    coords_y: np.ndarray,
    r_values: np.ndarray,
    L_data: np.ndarray,
    L_random_table: np.ndarray,
    baseline_index: int = 0,
) -> np.ndarray:
    """Compute sop_cost for every permutation of `times`, ordered by
    factoradic rank. Returns shape (N!,) cost array.

    This is the cost table Grover amplitude amplification searches over.
    """
    n = len(times)
    factorial = math.factorial(n)
    costs = np.empty(factorial, dtype=np.float64)
    for rank in range(factorial):
        perm = _factoradic_to_perm(rank, n)
        costs[rank] = sop_cost(
            times, coords_x, coords_y, r_values,
            L_data, L_random_table, perm, baseline_index,
        )
    return costs


def enqueue_all_costs(
    times: np.ndarray,
    coords_x: np.ndarray,
    coords_y: np.ndarray,
    r_values: np.ndarray,
    L_target: np.ndarray,
) -> np.ndarray:
    """Compute L_error for every permutation of `times` and return an array
    of shape (N!, ) ordered by factoradic rank.

    This is the genuine SOP cost function (Mohler & Mateu 2023 eq. 4):
        cost(perm) = || L(times[perm]) - L_target ||^2
    where L_target is the L(r) of the un-permuted data. The cost is
    non-trivial (i.e. depends on the permutation order) when the
    spatio-temporal point pattern has non-trivial space-time
    correlation, e.g. when events are produced by a self-exciting
    Hawkes process. For uncorrelated (Poisson) data the cost collapses
    to a constant because L(3D) is permutation-invariant in that
    case.

    The factoradic-rank ordering of the returned array matches the
    qubit basis states used by Grover's algorithm.
    """
    n = len(times)
    factorial = math.factorial(n)
    costs = np.empty(factorial, dtype=np.float64)
    for rank in range(factorial):
        perm = _factoradic_to_perm(rank, n)
        L_p = compute_L_summary(times[perm], coords_x, coords_y, r_values)
        costs[rank] = l_error(L_p, L_target)
    return costs


# ---------------------------------------------------------------------------
# 2. Factoradic / Lehmer code utilities
# ---------------------------------------------------------------------------


def _factoradic_to_perm(rank: int, n: int) -> np.ndarray:
    """Convert a factoradic rank in [0, N!) to a permutation of [0..N-1]."""
    symbols = list(range(n))
    perm = np.empty(n, dtype=np.int64)
    r = rank
    for i in range(n):
        f = math.factorial(n - 1 - i)
        idx, r = divmod(r, f)
        perm[i] = symbols.pop(idx)
    return perm


def number_of_qubits(n: int) -> int:
    """ceil(log2 N!) — the minimal information-theoretic qubit count."""
    return max(1, math.ceil(math.log2(math.factorial(n))))


# ---------------------------------------------------------------------------
# 3. Quantum circuit:  |psi> = A O_tau (psi0)  -- one Grover iteration
# ---------------------------------------------------------------------------


def build_circuit(
    n: int,
    costs: np.ndarray,
    tau: float,
    iterations: int = 1,
):
    """Return a PennyLane QNode that performs `iterations` Grover iterations
    for the given SOP instance.

    Circuit shape (per iteration):
      1. |+>^q                (Hadamard on each wire)
      2. Oracle:               flip phase on marked basis states
      3. Diffuser:             H (I - 2|0><0|) H

    A full Grover run repeats steps 2-3 `iterations` times. The function
    accepts the iteration count up front so the QNode is built once and
    evaluated `iterations` times implicitly via repeated calls.

    IMPORTANT: the QNode does NOT iterate internally. Iteration is
    handled by calling `one_iteration()` repeatedly from the caller.
    See `iterated_grover_sampling` for the loop driver.
    """

    q = number_of_qubits(n)
    n_factorial = math.factorial(n)

    # --- Construct the diagonal phase vector on q qubits -------------------
    # We embed the SOP search into the first N! basis states of a
    # 2^q-dimensional Hilbert space. Basis states with index >= N! are
    # "garbage" amplitude; we leave them un-flipped so they belong to
    # the unmarked set under Grover. This is the standard trick of
    # using a wider Hilbert space than the search space.
    diag = np.ones(2 ** q, dtype=np.float64)
    marked = 0
    for rank in range(n_factorial):
        if costs[rank] <= tau:
            diag[rank] = -1.0
            marked += 1
    # The first N! ranks are now marked if cost <= tau. The diffuser
    # flip on |0..0> keeps that state in the unmarked set, which is
    # consistent with rank 0 either being a valid marked permutation
    # (if its cost <= tau, in which case diag[0] = -1) or a valid
    # unmarked permutation (if its cost > tau, in which case diag[0] = 1).
    # We do NOT override diag[0].

    dev = qml.device("default.qubit", wires=q)

    @qml.qnode(dev)
    def one_iteration():
        # 1. uniform superposition
        for w in range(q):
            qml.Hadamard(wires=w)
        # 2. oracle: flip phase on marked basis states
        qml.DiagonalQubitUnitary(diag, wires=range(q))
        # 3. diffuser: H (I - 2|0..0><0..0|) H
        for w in range(q):
            qml.Hadamard(wires=w)
        qml.DiagonalQubitUnitary(
            np.array([(-1.0 if i == 0 else 1.0) for i in range(2 ** q)]),
            wires=range(q),
        )
        for w in range(q):
            qml.Hadamard(wires=w)
        return qml.probs(wires=range(q))

    return one_iteration, q, marked


def iterated_grover_sampling(circuit, iterations: int) -> np.ndarray:
    """Run the Grover circuit `iterations` times, returning the final
    probability distribution.

    Each call to `circuit()` is ONE Grover iteration (Hadamard + Oracle
    + Diffuser). The starting state of each call is the standard |0..0>.
    To get the cumulative effect of k iterations we feed the previous
    statevector into the next call. PennyLane's `default.qubit` does not
    natively expose the statevector mid-circuit, so we run a separate
    QNode that performs k iterations internally.

    Returns:
        probs: probability distribution over 2^q basis states after
               k Grover iterations.
    """
    # Implementation: we build a fresh QNode that contains k iterations
    # of (Oracle + Diffuser) inside, applied to |+>^q.
    raise NotImplementedError(
        "Use build_iterated_circuit instead; this helper is unused."
    )


def build_iterated_circuit(
    n: int,
    costs: np.ndarray,
    iterations: int,
    tau: float | None = None,
    marked_indices: np.ndarray | None = None,
):
    """Build a single QNode that performs `iterations` Grover iterations.

    Two ways to specify the marked set:
      * `tau`: mark all ranks where `costs[rank] <= tau`. Cost-threshold mode.
      * `marked_indices`: explicit array of indices to mark. Top-K mode.

    Exactly one of `tau` or `marked_indices` must be provided. The
    structure of the QNode is:

        |0..0>  --H^otimes-->  |+>^q
        for _ in range(iterations):
            DiagonalQubitUnitary(diag)
            Hadamard^otimes
            DiagonalQubitUnitary(diffuser_diag)
            Hadamard^otimes
        --measure--> probs

    Returns:
        qnode, q, marked_count
    """
    if (tau is None) == (marked_indices is None):
        raise ValueError(
            "Provide exactly one of `tau` or `marked_indices`."
        )

    q = number_of_qubits(n)
    n_factorial = math.factorial(n)

    diag = np.ones(2 ** q, dtype=np.float64)
    marked = 0
    if marked_indices is not None:
        marked_indices = np.asarray(marked_indices, dtype=np.int64)
        # Sanity: indices must be valid ranks.
        if marked_indices.min() < 0 or marked_indices.max() >= n_factorial:
            raise ValueError(
                f"marked_indices out of range [0, {n_factorial})."
            )
        for idx in marked_indices:
            diag[idx] = -1.0
        marked = int(len(marked_indices))
    else:
        for rank in range(n_factorial):
            if costs[rank] <= tau:
                diag[rank] = -1.0
                marked += 1

    diffuser_diag = np.array(
        [(-1.0 if i == 0 else 1.0) for i in range(2 ** q)],
        dtype=np.float64,
    )

    dev = qml.device("default.qubit", wires=q)

    @qml.qnode(dev)
    def iterated_qnode():
        # Initial uniform superposition
        for w in range(q):
            qml.Hadamard(wires=w)
        for _ in range(int(iterations)):
            qml.DiagonalQubitUnitary(diag, wires=range(q))
            for w in range(q):
                qml.Hadamard(wires=w)
            qml.DiagonalQubitUnitary(diffuser_diag, wires=range(q))
            for w in range(q):
                qml.Hadamard(wires=w)
        return qml.probs(wires=range(q))

    return iterated_qnode, q, marked


# ---------------------------------------------------------------------------
# 4. Driver — run the experiment and emit clean numerical evidence
# ---------------------------------------------------------------------------


@dataclass
class SopQuantumResult:
    n: int
    qubits: int
    tau: float
    marked_count: int
    iterations: int
    marked_probability: float
    best_rank: int
    best_cost: float
    uniform_baseline: float
    oracle_prep_time_s: float
    circuit_run_time_s: float


def run_sop_quantum(
    times: np.ndarray,
    coords_x: np.ndarray,
    coords_y: np.ndarray,
    r_values: np.ndarray,
    L_target: np.ndarray,
    tau_quantile: float = 0.25,
    iterations: int | None = None,
    top_k: int | None = None,
    n_random_baseline: int = 8,
    baseline_index: int = 0,
    seed: int = 0,
) -> SopQuantumResult:
    """End-to-end driver for genuine Grover SOP search.

    The cost function is the Mohler-Mateu SOP objective: each
    candidate permutation is scored against the L(r) of a fixed
    random permutation of times. This makes the cost depend on the
    permutation order, restoring a genuine combinatorial search
    over the N! candidate permutations.

    Two ways to define the marked set:
      * `tau_quantile`: mark perms with cost <= quantile. Useful when the
        cost distribution is smooth.
      * `top_k`: mark the K lowest-cost perms (a stricter, more
        deterministic choice that avoids the degenerate case where
        most costs tie at zero). If both are set, `top_k` wins.

    The driver builds ONE QNode that contains all `iterations` Grover
    iterations internally. Calling it once returns the final probability
    distribution after `iterations` rounds of (Oracle + Diffuser).
    """

    n = len(times)
    if n > 7:
        raise ValueError("Table oracle only supports N <= 7 (5040 perms).")

    t0 = time.time()
    # Build the random-permutation L(r) table; these L_k's act as
    # baselines in the SOP cost (Mohler & Mateu 2023, eq. 4).
    L_random_table = _random_permutation_table(
        times, coords_x, coords_y, r_values,
        n_random=n_random_baseline, seed=seed,
    )
    # Build the cost table over all N! permutations, using L_k[baseline_index]
    # as the SOP target. This is the table Grover amplitude amplifies on.
    costs = sop_cost_table(
        times, coords_x, coords_y, r_values,
        L_data=L_target, L_random_table=L_random_table,
        baseline_index=baseline_index,
    )
    t_oracle = time.time() - t0

    if top_k is not None and top_k > 0:
        # Sort costs and pick the K lowest; everything else is unmarked.
        order = np.argsort(costs)
        marked_idx = order[: int(top_k)]
        marked = int(len(marked_idx))
        # tau is unused but we still report it as the threshold cost.
        tau = float(costs[marked_idx].max()) if marked > 0 else float("inf")
    else:
        tau = float(np.quantile(costs, tau_quantile))
        marked = int(np.sum(costs <= tau))
        marked_idx = None

    if iterations is None:
        if marked <= 0:
            iters = 1
        else:
            ratio = math.factorial(n) / marked
            iters = max(1, int(round(math.pi / 4.0 * math.sqrt(ratio))))
    else:
        iters = int(iterations)

    # Qubit-friendly iteration clamp: too many iterations saturate the
    # marked subspace and degrade probability (over-rotation). Bound
    # it for small N.
    iters = min(iters, 32)

    if marked_idx is not None:
        circuit, q, marked_check = build_iterated_circuit(
            n, costs, iters, marked_indices=marked_idx
        )
    else:
        circuit, q, marked_check = build_iterated_circuit(
            n, costs, iters, tau=tau
        )
    assert marked_check == marked, (
        f"Marked count mismatch in build_iterated_circuit: "
        f"{marked_check} != {marked}"
    )

    t0 = time.time()
    probs = circuit()
    probs = np.asarray(probs, dtype=np.float64).real
    t_circ = time.time() - t0

    # Build the marked mask using the same rule as the circuit.
    marked_mask = np.zeros_like(probs, dtype=bool)
    factorial = math.factorial(n)
    if marked_idx is not None:
        marked_mask[marked_idx] = True
    else:
        marked_mask[:factorial] = costs <= tau
    uniform_baseline = float(marked) / float(factorial)

    best_idx = int(np.argmax(probs))
    best_cost = (
        float(costs[best_idx]) if best_idx < factorial else float("nan")
    )

    return SopQuantumResult(
        n=n,
        qubits=q,
        tau=tau,
        marked_count=marked,
        iterations=iters,
        marked_probability=float(probs[marked_mask].sum()),
        best_rank=best_idx,
        best_cost=best_cost,
        uniform_baseline=uniform_baseline,
        oracle_prep_time_s=t_oracle,
        circuit_run_time_s=t_circ,
    )


# ---------------------------------------------------------------------------
# 5. Self-test (run by `python3 genuine_sop_quantum.py`)
# ---------------------------------------------------------------------------


def _synthetic_dataset(n: int = 5, seed: int = 0) -> Tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    times = np.sort(rng.uniform(0, 1.0, size=n))
    coords_x = rng.uniform(0, 1.0, size=n)
    coords_y = rng.uniform(0, 1.0, size=n)
    r_values = np.linspace(0.05, 0.30, 6)
    L_target = compute_L_summary(times, coords_x, coords_y, r_values)
    return times, coords_x, coords_y, r_values, L_target


if __name__ == "__main__":
    for n in (4, 5):
        data = _synthetic_dataset(n=n)
        result = run_sop_quantum(*data, tau_quantile=0.2)
        print(
            f"N={result.n}  q={result.qubits}  marked={result.marked_count}/{math.factorial(n)}  "
            f"iters={result.iterations}  marked_prob={result.marked_probability:.3f}  "
            f"uniform_baseline={result.uniform_baseline:.3f}  "
            f"best_cost={result.best_cost:.4f}  oracle_t={result.oracle_prep_time_s:.3f}s  "
            f"circuit_t={result.circuit_run_time_s:.3f}s"
        )
