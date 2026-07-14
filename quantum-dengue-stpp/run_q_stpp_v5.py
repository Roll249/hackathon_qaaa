#!/usr/bin/env python3
"""
Q-STPP v5: SELECTED warm-start, NaN guards, fallback counter, n_qubits sweep.

Builds on v4 (which solves C1-C4 critical bugs) and adds:
  F1 [warm-start=None] - variants with warm_start=False (Finding #2 in code review)
  F2 [NaN guard]       - check classical_pred.mean() for NaN before warm-start (Finding #4)
  F3 [fallback count]  - expose silent quantum gradient degradation (Finding #3)
  F4 [n_qubits sweep]  - 6, 8, 12 (8 was the v4-reported sweet spot)
"""
import os, sys, json, time, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

# Re-use all v4 components except the QuantumIntensityGenerator (which we improve).
import numpy as np
import torch
import torch.nn as nn
import run_q_stpp_v4 as v4
# Names from v4 — we re-export them so existing test code keeps working.
generate_lgcp_clean = v4.generate_lgcp_clean
classical_embedding = v4.classical_embedding
apply_sop_swapnetwork = v4.apply_sop_swapnetwork
expand_with_covariance = v4.expand_with_covariance
ClassicalMLP = v4.ClassicalMLP
v4_RealSWAPNetworkQAOA = v4.RealSWAPNetworkQAOA

OUTPUT_DIR = 'output_result/q_stpp_v5'
os.makedirs(OUTPUT_DIR, exist_ok=True)
SEED = 42


# ============================================================================
# QuantumIntensityGeneratorV5 — improvements from code review
# ============================================================================

class QuantumIntensityGeneratorV5(nn.Module):
    def __init__(self, in_dim, n_qubits=8, n_layers=2, dev=None):
        super().__init__()
        self.in_dim = in_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.use_quantum = True
        self.fallback_count = 0  # F3: expose silent degradation

        try:
            import pennylane as qml
            self.qml = qml
            self.PENNYLANE_OK = True
        except ImportError:
            self.PENNYLANE_OK = False
            return

        self.proj = nn.Linear(in_dim, n_qubits)
        nn.init.normal_(self.proj.weight, std=0.1)
        nn.init.zeros_(self.proj.bias)

        self.theta = nn.Parameter(torch.randn(n_layers, n_qubits, 3) * 0.3)
        self.q_fc = nn.Linear(n_qubits, 1)
        # Default init: small non-zero (avoid the v4 zero-init trap)
        nn.init.normal_(self.q_fc.weight, std=0.05)
        nn.init.zeros_(self.q_fc.bias)

        dev = dev or qml.device('default.qubit', wires=n_qubits)

        @qml.qnode(dev, interface='autograd')
        def circuit(inputs, theta):
            for q in range(n_qubits):
                qml.Rot(inputs[q], inputs[q] * 0.5, inputs[q] * 0.3, wires=q)
            for L in range(n_layers):
                for q in range(n_qubits):
                    qml.Rot(theta[L, q, 0], theta[L, q, 1],
                            theta[L, q, 2], wires=q)
                for q in range(n_qubits):
                    qml.CZ(wires=[q, (q + 1) % n_qubits])
            return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]

        self.circuit = circuit

    def warm_start_from_classical(self, classical_model, train_features, train_target):
        """F1+F2: warm-start with NaN guard + small-nonzero FC weight init."""
        with torch.no_grad():
            classical_pred = classical_model(train_features).detach()
            target_mean = float(classical_pred.mean())
            if not np.isfinite(target_mean):
                print("    [WARN] classical pretraining NaN — using target_mean=0.0")
                target_mean = 0.0
            self.q_fc.bias.data = torch.tensor([target_mean], dtype=torch.float32)
            # Small non-zero init (NOT zero like v4) — Finding #2 fix
            nn.init.normal_(self.q_fc.weight, std=0.05)

    def forward(self, x):
        if not self.PENNYLANE_OK or not self.use_quantum:
            return self.q_fc(torch.zeros(x.shape[0], self.n_qubits)).squeeze(-1)
        x = torch.FloatTensor(x) if isinstance(x, np.ndarray) else x
        x_proj = torch.tanh(self.proj(x)) * np.pi
        batch_size = x.shape[0]
        z_out = torch.zeros(batch_size, self.n_qubits, dtype=torch.float32)
        for i in range(batch_size):
            try:
                z = self.circuit(x_proj[i].float(), self.theta)
                z_stack = torch.stack([z[q] for q in range(self.n_qubits)])
                z_out[i] = z_stack
            except Exception:
                self.fallback_count += 1
                z_out[i] = torch.zeros(self.n_qubits, dtype=torch.float32)
        return self.q_fc(z_out).squeeze(-1)


# ============================================================================
# train_eval (same body as v4 — correct eval slice indexing)
# ============================================================================

def train_eval(X, features, lam_true,
               use_sop=False, use_ent=False, use_intensity_q=False,
               n_qubits=8, n_epochs=40, lr=0.05,
               warm_start=True, seed=42, use_v5_quantum=True):
    """Note: defaults here are conservative; main() sets them explicitly.

    use_v5_quantum controls whether intensity_q uses v5 QuantumIntensityGenerator
    (with NaN guard, fallback counter, small-nonzero FC init) — vs v4's."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    features_emb = classical_embedding(X, features, n_clusters=4)
    X_p, features_p, lam_p = apply_sop_swapnetwork(
        X, features_emb, lam_true, use_quantum=use_sop, seed=seed
    )
    features_aug = expand_with_covariance(features_p, use_quantum=use_ent)
    in_dim = features_aug.shape[1]

    lam_mean = float(lam_p.mean())
    lam_std = float(lam_p.std() + 1e-6)
    lam_p_norm = (lam_p - lam_mean) / lam_std

    if use_intensity_q:
        QuantumCls = QuantumIntensityGeneratorV5 if use_v5_quantum else v4.QuantumIntensityGeneratorV4
        model = QuantumCls(in_dim=in_dim, n_qubits=n_qubits, n_layers=2)
        if warm_start and getattr(model, 'PENNYLANE_OK', False):
            classical = ClassicalMLP(in_dim=in_dim)
            opt_cl = torch.optim.Adam(classical.parameters(), lr=lr)
            n = len(X_p)
            n_train_inner = int(n * 0.7)
            f_tr_t = torch.FloatTensor(features_aug[:n_train_inner])
            l_tr_norm = torch.FloatTensor(lam_p_norm[:n_train_inner])
            for e in range(15):
                opt_cl.zero_grad()
                pred = classical(f_tr_t)
                loss = ((pred - l_tr_norm) ** 2).mean()
                loss.backward()
                opt_cl.step()
            if hasattr(model, 'warm_start_from_classical'):
                model.warm_start_from_classical(classical, f_tr_t, l_tr_norm)
    else:
        model = ClassicalMLP(in_dim=in_dim, hidden=64, depth=3)

    n = len(X_p)
    n_train = int(n * 0.7)
    f_train = features_aug[:n_train]
    f_test = features_aug[n_train:]
    lam_train_norm = lam_p_norm[:n_train]
    lam_test_norm = lam_p_norm[n_train:]

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    f_train_t = torch.FloatTensor(f_train)
    f_test_t = torch.FloatTensor(f_test)
    lam_train_t = torch.FloatTensor(lam_train_norm)
    lam_test_t = torch.FloatTensor(lam_test_norm)

    losses = []
    t0 = time.time()
    for epoch in range(n_epochs):
        if hasattr(model, 'train'):
            model.train()
        opt.zero_grad()
        pred = model(f_train_t)
        loss = ((pred - lam_train_t) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    train_time = time.time() - t0

    if hasattr(model, 'eval'):
        model.eval()
    with torch.no_grad():
        lam_pred_test_norm = model(f_test_t).numpy()
        lam_pred_train_norm = model(f_train_t).numpy()

    lam_pred_test = lam_pred_test_norm * lam_std + lam_mean
    lam_pred_train = lam_pred_train_norm * lam_std + lam_mean
    lam_test_orig = lam_p[n_train:]
    lam_train_orig = lam_p[:n_train]

    ss_res = float(np.sum((lam_test_orig - lam_pred_test) ** 2))
    ss_tot = float(np.sum((lam_test_orig - lam_test_orig.mean()) ** 2))
    r2 = float(1 - ss_res / (ss_tot + 1e-10))
    mae = float(np.mean(np.abs(lam_test_orig - lam_pred_test)))
    rmse = float(np.sqrt(np.mean((lam_test_orig - lam_pred_test) ** 2)))

    ss_res_tr = float(np.sum((lam_train_orig - lam_pred_train) ** 2))
    ss_tot_tr = float(np.sum((lam_train_orig - lam_train_orig.mean()) ** 2))
    r2_train = float(1 - ss_res_tr / (ss_tot_tr + 1e-10))

    fallback_count = getattr(model, 'fallback_count', 0)

    return {
        'r2_lambda': r2,
        'mae_lambda': mae,
        'rmse_lambda': rmse,
        'r2_train': r2_train,
        'final_loss': losses[-1],
        'best_loss': float(min(losses)),
        'n_train': n_train,
        'n_test': n - n_train,
        'in_dim': in_dim,
        'time': train_time,
        'n_qubits': n_qubits,
        'warm_start': warm_start,
        'fallback_count': fallback_count,
        'lam_pred_test': lam_pred_test.tolist(),
        'lam_test_orig': lam_test_orig.tolist(),
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║  Q-STPP v5: POST-REVIEW FIXES                                     ║
║  F1 warm_start=False  F2 NaN guard  F3 fallback counter  F4 nq sweep ║
╚══════════════════════════════════════════════════════════════════════╝
""")

    print("  [Setup] Generating LGCP dataset (n_reps=50, n_time=10)...")
    t0 = time.time()
    X, features, lam_true = generate_lgcp_clean(n_reps=50, n_time=10, grid_size=10)
    print(f"    X={X.shape}, features={features.shape}, lam_true={lam_true.shape}")
    print(f"    Time: {time.time() - t0:.1f}s")

    # Sanity check baseline (must match v4 R²=0.9717 if everything is correct)
    r_baseline = train_eval(X, features, lam_true,
        use_sop=False, use_ent=False, use_intensity_q=False,
        n_qubits=6, warm_start=True, n_epochs=30, seed=42,
        use_v5_quantum=False)
    print(f"    Sanity: CCC R²={r_baseline['r2_lambda']:+.4f} (target: +0.9717)")

    # Build sweep — start with sanity configs (4) then add new axes
    configs = []
    # === v4-style 8 configs (warm_start=True, n_qubits=6) using V5 quantum ===
    for sop in [False, True]:
        for ent in [False, True]:
            for fg in [False, True]:
                configs.append({
                    'use_sop': sop, 'use_ent': ent, 'use_intensity_q': fg,
                    'warm_start': True, 'n_qubits': 6, 'use_v5_quantum': True,
                    'label': f"QSOP={'Q' if sop else 'C'}_QE={'Q' if ent else 'C'}_QG={'Q' if fg else 'C'}_WS=T_nq=6",
                })

    # === F1: warm_start=False variants (NEW) ===
    # Quantum intensity generator only — compare T vs F for FG=T
    for ws in [False, True]:
        configs.append({
            'use_sop': False, 'use_ent': False, 'use_intensity_q': True,
            'warm_start': ws, 'n_qubits': 6, 'use_v5_quantum': True,
            'label': f"QSOP=C_QE=C_QG=Q_WS={'T' if ws else 'F'}_nq=6",
        })

    # === F4: n_qubits sweep with best config ===
    for nq in [8, 12]:
        for ws in [True, False]:
            configs.append({
                'use_sop': False, 'use_ent': True, 'use_intensity_q': True,
                'warm_start': ws, 'n_qubits': nq, 'use_v5_quantum': True,
                'label': f"QSOP=C_QE=Q_QG=Q_WS={'T' if ws else 'F'}_nq={nq}",
            })

    print(f"\n  Running {len(configs)} configurations...")
    results = []
    for i, cfg in enumerate(configs):
        print(f"\n  [{i+1}/{len(configs)}] {cfg['label']}...", flush=True)
        try:
            r = train_eval(X, features, lam_true,
                use_sop=cfg['use_sop'], use_ent=cfg['use_ent'],
                use_intensity_q=cfg['use_intensity_q'],
                n_qubits=cfg['n_qubits'], warm_start=cfg['warm_start'],
                n_epochs=30, seed=42, use_v5_quantum=cfg['use_v5_quantum'])
            r['config'] = cfg
            results.append(r)
            print(f"    → R²={r['r2_lambda']:+.4f}, MAE={r['mae_lambda']:.4f}, "
                  f"fallback={r['fallback_count']}, time={r['time']:.1f}s",
                  flush=True)
        except Exception as e:
            print(f"    FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append({'config': cfg, 'r2_lambda': -1.0,
                            'mae_lambda': 999, 'fallback_count': 0,
                            'time': 0.0, 'in_dim': 0})

    # Save compact
    compact = []
    for r in results:
        cr = {k: v for k, v in r.items()
              if k not in ('lam_pred_test', 'lam_test_orig')}
        compact.append(cr)
    with open(os.path.join(OUTPUT_DIR, 'q_stpp_v5_results.json'), 'w') as f:
        json.dump(compact, f, indent=2)

    # Summary
    baseline = next((r for r in results
                     if not r['config']['use_sop']
                     and not r['config']['use_ent']
                     and not r['config']['use_intensity_q']), results[0])

    print(f"\n{'='*80}")
    print(f"  ALL CONFIGURATIONS - v5")
    print(f"{'='*80}\n")
    print(f"{'#Q':>3} {'SOP':>5} {'Ent':>5} {'FG':>5} {'WS':>4} {'NQ':>3} "
          f"{'R²_λ':>9} {'MAE_λ':>9} {'FB':>5} {'Time':>7}")
    print('-' * 65)
    for r in results:
        c = r['config']
        n_q = sum([c['use_sop'], c['use_ent'], c['use_intensity_q']])
        print(f"{n_q:>3d} {'Q' if c['use_sop'] else 'C':>5} "
              f"{'Q' if c['use_ent'] else 'C':>5} "
              f"{'Q' if c['use_intensity_q'] else 'C':>5} "
              f"{'T' if c['warm_start'] else 'F':>4} "
              f"{c['n_qubits']:>3d} "
              f"{r['r2_lambda']:>+9.4f} {r['mae_lambda']:>9.4f} "
              f"{r.get('fallback_count', 0):>5} {r['time']:>6.1f}s")

    best = max(results, key=lambda x: x.get('r2_lambda', -999))
    print(f"\n  BASELINE (CCC): R²={baseline['r2_lambda']:+.4f}, MAE={baseline['mae_lambda']:.4f}")
    print(f"  BEST OVERALL : R²={best['r2_lambda']:+.4f}, MAE={best['mae_lambda']:.4f}")
    print(f"  CONFIG       : {best['config']['label']}")
    print(f"  WINNER       : {'QUANTUM' if best['r2_lambda'] > baseline['r2_lambda'] else 'BASELINE'}")


if __name__ == '__main__':
    main()