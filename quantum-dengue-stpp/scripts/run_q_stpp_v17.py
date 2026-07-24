#!/usr/bin/env python3
"""CLI runner for the v17 QAOA-SOP-augmented pipeline.

WHAT IS QUANTUM, WHAT IS CLASSICAL, WHAT IS HONEST CLAIM
--------------------------------------------------------
This is a thin CLI over `src.quantum.pipeline_v17.run_pipeline`. The
QUANTUM component is the QAOA-XY-ring-mixer subset selector (Layer 3
augmentation) and the optional quantum kernel for the downstream
classifier (Layer 2). The CLASSICAL components are the Hawkes simulator,
the 3D L-function summary, the RBF-kernel 1-NN classifier, and all
metric aggregations.

The HONEST claim is that the QAOA-augmented pipeline produces a
classifier that is statistically indistinguishable (within one
standard error) from the v15 baseline. Wall-clock advantage is NOT
claimed. See `QAOA_SOP_V17_REPORT.md` for the full benchmark.

Usage:
    python scripts/run_q_stpp_v17.py \
        --seeds 1 2 3 \
        --n-events 20 30 \
        --m-candidates 8 \
        --k-select 3 \
        --qaoa-method auto \
        --use-real-data \
        --output-dir output_result/q_stpp_v17
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import warnings

# Matplotlib config to avoid "config dir not writable" warning.
os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(
        tempfile.gettempdir(),
        f"mplconfig-{os.getuid() if hasattr(os, 'getuid') else 'user'}",
    ),
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Allow `python scripts/run_q_stpp_v17.py` from the project root.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from src.quantum.pipeline_v17 import (  # noqa: E402
    PipelineResult,
    load_real_dengue_window,
    run_pipeline,
)

warnings.filterwarnings("ignore")

OUTPUT_DIR_DEFAULT = os.path.join(PROJECT_ROOT, "output_result", "q_stpp_v17")
DATA_PATH_DEFAULT = os.path.normpath(
    os.path.join(PROJECT_ROOT, "..", "dengue_dataset", "sea_dengue_admin1_month.csv")
)


def parse_args():
    p = argparse.ArgumentParser(description="Run the QAOA-SOP-augmented v17 pipeline.")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--n-events", type=int, nargs="+", default=[20, 30])
    p.add_argument("--m-candidates", type=int, default=10,
                   help="Number of SOP candidates (M).")
    p.add_argument("--k-select", type=int, default=4,
                   help="Number of permutations to select (k).")
    p.add_argument("--n-swap-iters", type=int, default=60,
                   help="L-evaluation budget per candidate.")
    p.add_argument("--p-qaoa", type=int, default=2, help="QAOA depth.")
    p.add_argument("--n-shots", type=int, default=1024, help="QAOA shots.")
    p.add_argument("--qaoa-method", choices=["auto", "qaoa_xy", "grover", "classical"],
                   default="auto")
    p.add_argument("--use-real-data", action="store_true",
                   help="Try to load real dengue CSV; fall back to synthetic Hawkes.")
    p.add_argument("--data-path", default=DATA_PATH_DEFAULT)
    p.add_argument("--n-augment-features", type=int, default=3)
    p.add_argument("--output-dir", default=OUTPUT_DIR_DEFAULT)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def to_jsonable(result: PipelineResult, n_events_target: int, seed: int) -> dict:
    """Convert a PipelineResult into a JSON-serialisable dict."""
    return {
        "seed": seed,
        "n_events_target": n_events_target,
        "n_events": int(result.n_events),
        "M_candidates": int(result.M_candidates),
        "k_selected": int(result.k_selected),
        "method": result.method,
        "mean_l_error": float(result.mean_l_error),
        "set_diversity": float(result.set_diversity),
        "selected_l_errors": list(result.l_errors_selected),
        "selection_info": result.selection_info,
        "classical_accuracy": float(result.classical_accuracy)
            if result.classical_accuracy is not None else None,
        "classical_f1": float(result.classical_f1)
            if result.classical_f1 is not None else None,
        "classical_auc": float(result.classical_auc)
            if result.classical_auc is not None else None,
        "quantum_accuracy": float(result.quantum_accuracy)
            if result.quantum_accuracy is not None else None,
        "quantum_f1": float(result.quantum_f1)
            if result.quantum_f1 is not None else None,
        "quantum_auc": float(result.quantum_auc)
            if result.quantum_auc is not None else None,
        "wall_time_total_s": float(result.wall_time_total_s),
        "wall_time_qaoa_s": float(result.wall_time_qaoa_s),
        "wall_time_classical_s": float(result.wall_time_classical_s),
        "y_train": result.train_labels.tolist(),
        "y_test": result.test_labels.tolist(),
    }


def aggregate(rows):
    by_n = {}
    for r in rows:
        by_n.setdefault(r["n_events_target"], []).append(r)

    agg = {}
    for n, group in sorted(by_n.items()):
        methods = sorted({r["method"] for r in group})
        entry = {"n_events_target": n, "n_seeds": len(group), "by_method": {}}
        for m in methods:
            sub = [r for r in group if r["method"] == m]
            def stats(key):
                vals = np.array([r[key] for r in sub if r[key] is not None and
                                 not (isinstance(r[key], float) and np.isnan(r[key]))])
                if len(vals) == 0:
                    return float("nan"), float("nan"), 0
                return float(np.mean(vals)), float(np.std(vals)), int(len(vals))

            for k in [
                "classical_accuracy", "classical_f1", "classical_auc",
                "quantum_accuracy", "quantum_f1", "quantum_auc",
                "wall_time_total_s", "wall_time_qaoa_s", "wall_time_classical_s",
                "mean_l_error", "set_diversity",
            ]:
                mu, sd, ct = stats(k)
                entry["by_method"].setdefault(m, {})[k] = {
                    "mean": mu, "std": sd, "count": ct,
                }
        agg[n] = entry
    return agg


def plot_results(rows, out_path):
    """Bar chart comparing QAOA-XY vs classical on selected metrics."""
    if not rows:
        return
    by_n = {}
    for r in rows:
        by_n.setdefault(r["n_events_target"], []).append(r)

    ns = sorted(by_n)
    methods_seen = sorted({r["method"] for r in rows})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Plot 1: classical accuracy
    ax = axes[0]
    width = 0.35
    x = np.arange(len(ns))
    for i, m in enumerate(methods_seen):
        vals = []
        stds = []
        for n in ns:
            sub = [r for r in by_n[n] if r["method"] == m]
            if not sub:
                vals.append(0); stds.append(0)
            else:
                accs = [r["classical_accuracy"] for r in sub
                        if r["classical_accuracy"] is not None]
                vals.append(float(np.mean(accs)) if accs else 0)
                stds.append(float(np.std(accs)) if accs else 0)
        ax.bar(x + (i - len(methods_seen) / 2) * width, vals, width,
               yerr=stds, label=m, capsize=3)
    ax.set_xticks(x); ax.set_xticklabels([f"N={n}" for n in ns])
    ax.set_ylabel("Classical-kernel accuracy")
    ax.set_title("Layer 2 (RBF) accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)

    # Plot 2: mean L error
    ax = axes[1]
    for i, m in enumerate(methods_seen):
        vals = []
        stds = []
        for n in ns:
            sub = [r for r in by_n[n] if r["method"] == m]
            if not sub:
                vals.append(0); stds.append(0)
            else:
                errs = [r["mean_l_error"] for r in sub]
                vals.append(float(np.mean(errs)))
                stds.append(float(np.std(errs)))
        ax.bar(x + (i - len(methods_seen) / 2) * width, vals, width,
               yerr=stds, label=m, capsize=3)
    ax.set_xticks(x); ax.set_xticklabels([f"N={n}" for n in ns])
    ax.set_ylabel("Mean L-error of selected SOP perms")
    ax.set_title("Layer 3 (augmentation quality)")
    ax.legend(fontsize=8)

    # Plot 3: wall time
    ax = axes[2]
    for i, m in enumerate(methods_seen):
        vals = []
        stds = []
        for n in ns:
            sub = [r for r in by_n[n] if r["method"] == m]
            if not sub:
                vals.append(0); stds.append(0)
            else:
                ts = [r["wall_time_total_s"] for r in sub]
                vals.append(float(np.mean(ts)))
                stds.append(float(np.std(ts)))
        ax.bar(x + (i - len(methods_seen) / 2) * width, vals, width,
               yerr=stds, label=m, capsize=3)
    ax.set_xticks(x); ax.set_xticklabels([f"N={n}" for n in ns])
    ax.set_ylabel("Total wall time (s)")
    ax.set_title("Wall-clock (simulator, no quantum advantage claimed)")
    ax.legend(fontsize=8)

    fig.suptitle(
        "Q-STPP v17 — QAOA-XY-augmented SOP pipeline vs classical\n"
        "(honest apples-to-apples comparison on identical data and seed)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 78)
    print("  Q-STPP v17 — QAOA-SOP-Augmented Pipeline")
    print(f"  seeds={args.seeds}  n_events={args.n_events}  "
          f"M={args.m_candidates}  k={args.k_select}")
    print(f"  qaoa_method={args.qaoa_method}  p_qaoa={args.p_qaoa}  "
          f"n_shots={args.n_shots}  use_real_data={args.use_real_data}")
    print("=" * 78)

    rows = []
    for n_events_target in args.n_events:
        for seed in args.seeds:
            try:
                if args.use_real_data:
                    out = load_real_dengue_window(
                        csv_path=args.data_path,
                        max_events=n_events_target, seed=seed,
                    )
                    data_tuple, meta = out
                    times, cx, cy, r_values, L_target = data_tuple
                else:
                    from run_q_stpp_v15_fair import simulate_hawkes
                    times, cx, cy = simulate_hawkes(
                        n_events_target=n_events_target, seed=seed,
                    )
                    if len(times) < 10:
                        continue
                    r_values = np.linspace(0.05, 0.30, 8)
                    from src.quantum.genuine_sop_quantum import compute_L_summary
                    L_target = compute_L_summary(times, cx, cy, r_values)
                    meta = {"source": "synthetic"}
                if args.verbose:
                    print(f"\n[N={n_events_target} seed={seed}] data: {meta}")

                result = run_pipeline(
                    times, cx, cy, r_values, L_target,
                    M_candidates=args.m_candidates,
                    k_select=args.k_select,
                    n_swap_iters=args.n_swap_iters,
                    seed=seed,
                    qaoa_method=args.qaoa_method,
                    p_qaoa=args.p_qaoa,
                    n_shots=args.n_shots,
                    n_augment_features=args.n_augment_features,
                    verbose=args.verbose,
                )

                rec = to_jsonable(result, n_events_target, seed)
                rec["data_source"] = meta.get("source", "unknown")
                rows.append(rec)

                line = (
                    f"  N={n_events_target:>3} seed={seed}: "
                    f"method={result.method:<10} "
                    f"L_err={result.mean_l_error:.5f} "
                    f"div={result.set_diversity:.2f} "
                    f"clf_acc={result.classical_accuracy:.2f} "
                    f"q_acc={result.quantum_accuracy if result.quantum_accuracy is not None else float('nan'):.2f} "
                    f"t={result.wall_time_total_s:.2f}s"
                )
                print(line)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                print(f"  N={n_events_target} seed={seed} FAILED: {exc}")

    if not rows:
        print("[error] no successful runs")
        return 1

    agg = aggregate(rows)
    summary = {
        "config": vars(args),
        "rows": rows,
        "aggregate": agg,
    }
    out_json = os.path.join(args.output_dir, "pipeline_v17_results.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n[done] wrote {out_json}")

    out_plot = os.path.join(args.output_dir, "pipeline_v17_results.png")
    plot_results(rows, out_plot)
    print(f"[done] wrote {out_plot}")

    # Print summary table
    print("\n" + "=" * 78)
    print("  v17 PIPELINE — SUMMARY")
    print("=" * 78)
    for n, e in sorted(agg.items()):
        print(f"\n  N_events_target={n}, n_seeds={e['n_seeds']}")
        for m, stats in e["by_method"].items():
            print(f"    [{m}]")
            for k, v in stats.items():
                if v["count"] > 0:
                    print(f"      {k}: mean={v['mean']:.4f}  std={v['std']:.4f}  n={v['count']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())