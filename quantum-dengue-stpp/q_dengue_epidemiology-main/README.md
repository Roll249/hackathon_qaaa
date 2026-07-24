# Q-Dengue Epidemiology: A Paradigm Shift

> *"Bắt cá leo cây"* — Dùng **quantum walk search** để tìm index case (source) trên commute graph,
> thay vì cố predict epidemic propagation (bất khả do chaos theory).

## Hai phần của project

### 1a. Quantum Walk Search (⚡ Headline result — empirical)

Dùng coined quantum walk $U = S \cdot C$ trên weighted directed graph:

```
U_walk = Shift @ GroverCoin
U_search = R_marked @ U_walk
```

Kết quả: Ring đạt P(marked)=1.0, sparse đạt P(marked)=0.85. Speedup 4–22× so với classical hitting time.
Điện Biên realistic graph: P(marked)=0.033 — no resonance (negative result thật).
Kaohsiung (dữ liệu thật, GPS thật, Taiwan CDC, N=329): P(marked)≈0.0001 — vẫn no resonance,
kể cả trên graph đô thị dày, đúng bán kính phát tán muỗi Aedes. Negative result có vẻ robust
across cả địa hình nông thôn synthetic lẫn đô thị thật.

### 1b. Grover Amplification Scaffold (⚠️ Classical simulation — scaffold cho QPU)

Grover amplification trên QPIE-encoded statevector để detect top-K.
Đây là classical simulation, chạy trên CPU. Scaffold cho khi có QPU.
Kết quả được đo bằng oracle counter thật.

---

## Benchmark results

### Quantum Walk — empirical measurements

```
python benchmarks/bench_weighted_walk.py
python benchmarks/probe_walk.py
```

| Graph | N | P(marked) peak | Speedup |
|-------|---|----------------|---------|
| Ring | 48 | 1.0000 | 16.3× |
| Grid | 48 | 0.2408 | 10.6× |
| Sparse binary | 48 | 0.2259 | 19.0× |
| Sparse weighted | 48 | 0.1165 | 12.7× |
| **Điện Biên** (reachable) | 130 | 0.033 | **NO** |
| **Kaohsiung** (real GPS, Taiwan CDC) | 329 | 0.0001 | **NO** |

### Grover Amplification — measured oracle queries

```
python benchmarks/bench_quantum_vs_classical.py
```

```
   N |  mean DH/Cl | match% | mean iters
-----|-------------|--------|------------
  16 |       3.60  |  100%  |        3.3
  32 |       4.52  |  100%  |        5.0
  64 |       4.46  |   67%  |        7.3
 128 |       5.13  |  100%  |       15.3
```

**DH/Cl > 1** → Dürr-Høyer max-finding dùng NHIỀU oracle query hơn classical scan (đúng lý thuyết:
với M>1 marked state, mỗi Grover iteration tốn √(N/M) query, cộng dồn O(N) — không có speedup cho
max-finding kiểu này, khác với min/search với M=1).
**Match rate < 100%** → phép đo lượng tử là random sample theo |amplitude|², không phải argmax tất định
— nên occasionally miss true max trên một lần thử.

> Bảng trên đã fix 2 bug trong `durr_hoyer_max.py` (2026-07-23): (1) phép đo dùng
> `rng.choice(p=probs)` thay vì `argmax` tất định — vì mọi marked-state có amplitude giống hệt nhau,
> `argmax` luôn trả về index nhỏ nhất một cách sai lệch; (2) tính M từ `oracle.values` giờ tính vào
> oracle-query count (trước đây đọc "miễn phí", làm tỉ lệ DH/Classical tốt giả tạo).

## Cài đặt

```bash
cd "$(dirname "$0")"

# Quantum walk (headline)
python benchmarks/bench_weighted_walk.py
python benchmarks/probe_walk.py

# Grover scaffold (measured)
python benchmarks/bench_quantum_vs_classical.py

# Full pipeline
python -c "from src.pipeline import run_full_pipeline; run_full_pipeline()"
```

## Files

```
q_dengue_epidemiology/
├── src/
│   ├── graph_dien_bien.py       # Classical GIS
│   ├── qpie_encoder.py          # QPIE state preparation
│   ├── durr_hoyer_max.py        # Grover + oracle counter
│   ├── lackadaisical_walk.py    # Multi-hotspot Grover
│   └── pipeline.py              # Orchestrator
├── benchmarks/
│   ├── bench_weighted_walk.py      # Quantum walk headline
│   ├── probe_walk.py               # Probe resonance
│   ├── bench_empirical_walk.py     # Empirical hitting time
│   ├── bench_quantum_vs_classical.py  # Measured Grover queries
│   └── visualize_pipeline.py
└── output/
    ├── weighted_walk_benchmark.json
    └── durr_hoyer_benchmark.json
```

## Reference

- Dürr & Høyer 1996: "A Quantum Algorithm for Finding the Minimum"
- Szegedy 2004: "Quantum speed-up of Markov chain based algorithms"
- Wong 2015: "Equivalence of Szegedy's and coined quantum walks"
