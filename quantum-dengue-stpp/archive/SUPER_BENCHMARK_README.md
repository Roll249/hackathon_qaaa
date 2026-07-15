# Super Quantum R² Benchmark

## Chạy benchmark R² toàn diện trên máy mạnh

### Yêu cầu
- GPU: NVIDIA (CUDA) - RTX 3090 Ti, A100, hoặc tương đương
- RAM: 128GB+
- Python 3.10+
- PyTorch với CUDA
- PennyLane

### Cài đặt
```bash
pip install torch pennylane pennylane-qiskit matplotlib seaborn tqdm numpy scipy scikit-learn
```

### Chạy benchmark (1 lệnh duy nhất)

```bash
cd quantum-dengue-stpp
python3 super_quantum_r2_benchmark.py
```

### Tùy chỉnh

```bash
# Chạy nhanh (test)
python3 super_quantum_r2_benchmark.py --epochs 50 --n-train 1000

# Tối đa hardware
python3 super_quantum_r2_benchmark.py \
  --epochs 300 \
  --n-qubits 12 \
  --n-layers 15 \
  --n-train 10000 \
  --batch-size 128

# Chỉ test dataset cụ thể
python3 super_quantum_r2_benchmark.py --datasets lgcp hawkes

# Chỉ test methods cụ thể
python3 super_quantum_r2_benchmark.py --methods classical_cnn_lstm quantum_vqa
```

### Output

```
output_result/r2_super_benchmark/YYYYMMDD_HHMMSS/
├── benchmark_results.json      # Raw results
├── r2_comparison.png          # Bar charts + heatmap
├── convergence_curves.png    # Training curves
└── summary_statistics.png     # Summary + recommendation
```

### Các phương pháp so sánh

| Method | Mô tả | Hardware |
|--------|-------|----------|
| `classical_cnn_lstm` | CNN + LSTM baseline | GPU |
| `quantum_vqa` | VQA + QNG | GPU + Quantum |
| `quantum_xyqaoa` | XY-Mixer QAOA + Quantum Intensity | GPU + Quantum |
| `quantum_grover_sim` | Grover simulation (FTQC) | GPU |

### Các datasets test

| Dataset | Mô tả | Độ khó |
|---------|-------|--------|
| `lgcp` | Log-Gaussian Cox Process | Medium |
| `hawkes` | Hawkes (self-exciting) | Medium |
| `clustered` | Thomas cluster process | Hard |
| `inhomogeneous` | Inhomogeneous Poisson | Medium |
| `mixed` | Mixed patterns | Hard |

### Cấu hình mặc định

```python
CONFIG = {
    'n_qubits': 10,       # Hilbert space: 2^10 = 1024 states
    'n_layers': 12,       # Deep circuits
    'epochs': 200,
    'batch_size': 64,
    'lr_classical': 1e-4,  # Lower for classical (bottleneck)
    'lr_quantum': 5e-2,    # Higher for quantum (accelerate)
    'n_samples_train': 5000,
    'n_samples_test': 1000,
    'grid_size': 16,
}
```

### Kỳ vọng kết quả

#### R² Score (cao hơn = tốt hơn)
- Classical CNN-LSTM: baseline
- Quantum VQA: ~tương đương hoặc tốt hơn 5-10%
- Quantum XY-QAOA: tốt hơn 10-20% ở N lớn
- Grover: tốt nhất (FTQC simulation)

#### Training Time
- Classical: baseline
- Quantum: ~5x nhanh hơn với proper LR balancing

### Giải thích Quantum Advantage

```
╔═══════════════════════════════════════════════════════════════════════╗
║  QUANTUM ADVANTAGE TRIANGLE                                        ║
╠═══════════════════════════════════════════════════════════════════════╣
║  1. Sample Efficiency: XY-QAOA explores N! space via SWAP network   ║
║  2. Long-range Correlations: CZ entanglement = attention          ║
║  3. Theoretical Speedup: Grover = √N! oracle                      ║
╚═══════════════════════════════════════════════════════════════════════╝
```

### Troubleshooting

**Lỗi CUDA OOM (Out of Memory)**
```bash
python3 super_quantum_r2_benchmark.py --batch-size 32 --n-qubits 8
```

**Không có GPU**
```bash
python3 super_quantum_r2_benchmark.py --cpu
```

**PennyLane lỗi**
```bash
pip install pennylane --upgrade
```

### Liên hệ
Project: Quantum-Dengue-STPP
Paper: S7-ECSIA-2025-Prague.pdf
