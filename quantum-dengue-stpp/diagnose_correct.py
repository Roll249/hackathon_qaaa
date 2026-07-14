#!/usr/bin/env python3
"""Corrected: test H1, H2 with proper slice indexing."""
import sys, time, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
import numpy as np
import torch
torch.manual_seed(42); np.random.seed(42)
import importlib
import run_q_stpp_v4 as v4
importlib.reload(v4)
from run_q_stpp_v4 import (
    generate_lgcp_clean, classical_embedding, expand_with_covariance,
    ClassicalMLP, QuantumIntensityGeneratorV4,
)


def eval_r2(f_te, model, lam_test, lam_mean, lam_std):
    """Proper R² evaluation."""
    model.eval()
    with torch.no_grad():
        p_te = model(f_te).numpy() * lam_std + lam_mean
    ss_res = float(np.sum((lam_test - p_te) ** 2))
    ss_tot = float(np.sum((lam_test - lam_test.mean()) ** 2))
    if ss_tot < 1e-10:
        return -1e10
    return 1 - ss_res / ss_tot


# Setup ONCE
X, features, lam_true = generate_lgcp_clean(n_reps=50, n_time=10, grid_size=10)
f_emb = classical_embedding(X, features)
f_aug = expand_with_covariance(f_emb, use_quantum=True)
n_tr = int(len(X) * 0.7)
f_tr = torch.FloatTensor(f_aug[:n_tr])
f_te = torch.FloatTensor(f_aug[n_tr:])
lam_mean, lam_std = lam_true.mean(), lam_true.std()
l_tr = torch.FloatTensor((lam_true[:n_tr] - lam_mean) / (lam_std + 1e-6))
lam_test = lam_true[n_tr:]  # PROPER SLICE

print(f"n_train={n_tr}, n_test={len(lam_test)}")
print(f"lam_test mean={lam_test.mean():.3f}, std={lam_test.std():.3f}")
print(f"\n{'Config':>30} {'R²':>8} {'InitL':>9} {'FinalL':>9} {'Time':>6}")
print('-' * 70)

# === Baseline CCC ===
torch.manual_seed(42); np.random.seed(42)
model = ClassicalMLP(in_dim=f_aug.shape[1])
init_loss = float(((model(f_tr) - l_tr) ** 2).mean())
opt = torch.optim.Adam(model.parameters(), lr=0.05)
t0 = time.time()
for ep in range(30):
    opt.zero_grad()
    loss = ((model(f_tr) - l_tr) ** 2).mean()
    loss.backward(); opt.step()
r2 = eval_r2(f_te, model, lam_test, lam_mean, lam_std)
print(f"{'CCC (baseline)':>30} {r2:>+8.4f} {init_loss:>9.4f} {loss.item():>9.4f} {time.time()-t0:>5.1f}s")

# === Quantum: with warm-start vs no warm-start, varied n_qubits ===
for nq in [4, 6, 8, 12]:
    for warm in [True, False]:
        torch.manual_seed(42); np.random.seed(42)
        model = QuantumIntensityGeneratorV4(in_dim=f_aug.shape[1], n_qubits=nq, n_layers=2)
        if warm:
            classical = ClassicalMLP(in_dim=f_aug.shape[1])
            opt_c = torch.optim.Adam(classical.parameters(), lr=0.05)
            for e in range(10):
                opt_c.zero_grad()
                loss_c = ((classical(f_tr) - l_tr) ** 2).mean()
                loss_c.backward(); opt_c.step()
            model.warm_start_from_classical(classical, f_tr, l_tr)
        init_loss = float(((model(f_tr) - l_tr) ** 2).mean())
        opt = torch.optim.Adam(model.parameters(), lr=0.05)
        t0 = time.time()
        for ep in range(30):
            opt.zero_grad()
            pred = model(f_tr)
            loss = ((pred - l_tr) ** 2).mean()
            loss.backward(); opt.step()
        r2 = eval_r2(f_te, model, lam_test, lam_mean, lam_std)
        label = f"CCQ nq={nq} warm={warm}"
        print(f"{label:>30} {r2:>+8.4f} {init_loss:>9.4f} {loss.item():>9.4f} {time.time()-t0:>5.1f}s", flush=True)