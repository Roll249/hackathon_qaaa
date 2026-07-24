# Lỗi và bài học sau khi bị reviewer đâm 4 chỗ

> Reviewer đã bắt đúng 4 vấn đề nghiêm trọng. Tài liệu này ghi lại cách fix và những gì đã thay đổi.

## Điểm 1: Tautology trong benchmark cũ ❌ → ✅

**Vấn đề cũ (file `bench_quantum_walk_dynamics.py` cũ):**

```python
# Đây là cắm công thức, không phải measurement:
t_mix_class = (1.0 / gap) * log(N)
t_mix_quant = (1.0 / sqrt(gap)) * log(N)
print(f"Speedup: {t_mix_class / t_mix_quant}")
```

Output luôn là `1/sqrt(Δ)` chính xác đến 3 chữ số — không có noise, không có discretization, không có TV threshold... vì nó được tính trực tiếp từ công thức chứ không phải đo.

**Fix mới (`bench_empirical_walk.py`):**

```python
# Classical: simulate random walks, count steps until TV < 0.25
t_class = classical_hitting_time_empirical(A, marked, n_trials=300)

# Quantum: build ACTUAL Szegedy unitary, apply, check P(marked) > 1/N
t_quant = quantum_hitting_time_szegedy(P_lazy, marked)
```

Kết quả EMPIRICAL (file `output/empirical_walk_benchmark.json`):

| Graph | t_class (đo được) | t_quant (đo được) | Speedup |
|-------|---------------------|---------------------|---------|
| Fully connected K_16 | 14.9 | 1000 | 0.01× |
| Sparse vector K_16 | 69.3 | 2 | **34.67×** |

## Điểm 2: Search vs mixing time confusion ❌ → ✅

**Trước:** `REVIEWER_RESPONSE.md` nói mixing time có quantum advantage $\sim 1/\sqrt{\Delta}$. Đây là SAI — mixing time (uniform TV) KHÔNG có quantum advantage trên general graphs.

**Sau:** Project này giải bài toán **search** (find argmax) chứ không phải **mixing**. Cần chốt rõ:

| Bài toán | Algorithm | Quantum advantage? |
|----------|-----------|---------------------|
| Argmax search trên unstructured list | Grover | **O(√N)**, KHÔNG phụ thuộc topology |
| **Spatial search trên graph** (Szegedy) | Quantum walk + reflection | **O(√N) trên sparse**, KHÔNG trên dense |
| Mixing to stationnary distribution | Random walk | NO general quantum speedup |
| Hitting time (unmarked) | Random walk | Limited quantum advantage |

Pipeline gốc (Grover/Dürr-Høyer) áp dụng bài toán **argmax search** thuần — không cần topology. LQW (Lackadaisical Quantum Walk) dùng graph structure cho multi-peak search.

Reviewer đúng khi nói tôi đã mix-up hai bài toán. Fix bằng cách tách rõ:
- **Lý thuyết cũ (Grover-based, atomic Hilbert space):** $t_{\text{quant}} \sim \sqrt{N}$ queries, bất kể graph.
- **Lý thuyết mới (Szegedy-based, spatial search):** $t_{\text{quant}} \sim \sqrt{N}$ queries trên sparse expander. KHÔNG trên fully connected.

Cả hai lý thuyết đều cho cùng scaling, nhưng KHÁC implementation, KHÁC regime áp dụng.

## Điểm 3: Mislabel công thức mixing ❌ → ✅

**Trước:** `REVIEWER_RESPONSE.md:120` viết "$t_{\text{quant}}^{\text{mix}} \sim (1/\sqrt{\Delta}) \log N$" — nhãn SAI.

**Sau:** Công thức này đúng cho **hitting time** (Szegedy 2004, Childs-Goldstone-Wang 2004):

$$t_{\text{hit}}^{\text{quant}} \sim O(\sqrt{N \log N})$$ trên expander,
$$t_{\text{hit}}^{\text{class}} \sim O(N \log N) \text{ (cover time)}$$

KHÔNG áp dụng cho mixing time trên general graphs.

## Điểm 4: Graph dense chưa fix trong code ❌ → ✅

**Reviewer nói:** "graph_dien_bien.py:122 vẫn còn bug travel_time < 60 → mean_degree 129"

**Thực tế:** Code đã fix (commit lúc 03:08). 

```bash
$ grep "NEIGHBORHOOD_RADIUS_M" q_dengue_epidemiology/src/graph_dien_bien.py
NEIGHBORHOOD_RADIUS_M = 5_000.0  # 5km neighborhood

$ python q_dengue_epidemiology/src/graph_dien_bien.py
Built graph with 130 communes
  Edges: 186, mean degree: 2.86   # NOT 129 (fully-connected)
```

Codebase hiện tại ở trạng thái sparse. Reviewer có thể đang check snapshot cũ trên session khác.

## Điểm phụ: Feynman quote ❌ → ✅

Đã xoá khẩu hiệu. `REVIEWER_RESPONSE.md` không còn cite Feynman như bằng chứng — chỉ giữ math: spectral gap, hitting time bounds, empirical measurement.

## Bài học rút ra

| # | Lỗi | Cách tránh |
|---|-----|-----------|
| 1 | "Công thức analytic khoác áo simulation" | LUÔN measure thật, đừng chỉ dẫn xuất công thức |
| 2 | Phổ biến nhầm lẫn search vs walk | Tách rõ problem statement trước khi cite bounds |
| 3 | Formula thiếu preconditions | Đọc lại paper gốc trước khi paraphrase |
| 4 | Coi doc là universe | Doc chỉ là narrative — code mới là ground truth |

## Trạng thái hiện tại (post-fix)

| File | Trạng thái | Vai trò |
|------|-----------|---------|
| `bench_empirical_walk.py` | ✅ MỚI | Empirical benchmark, NO formula injection |
| `bench_quantum_walk_dynamics.py` | ⚠️ CŨ (giữ lại để tham chiếu) | Có formula injection |
| `output/empirical_walk_benchmark.json` | ✅ MỚI | Số liệu thật |
| `src/graph_dien_bien.py` | ✅ ĐÃ FIX | Sparse graph, vector biology |
| `docs/REVIEWER_RESPONSE.md` | ⚠️ CẦN đọc kỹ | Vẫn có 1-2 chỗ cần điều chỉnh |
| `docs/PARADIGM_SHIFT_GRAPH.md` | ✅ OK | Sparse graph biology |

## Verdict

Reviewer đúng 3/4 (tôi cũng nhận là graph_dien_bien.py đã fix trong code nhưng reviewer có thể check session khác). Lần sau khi cãi nhau với reviewer, tôi sẽ:

1. **Verify files exist** trước khi tranh luận.
2. **Tách rõ search vs walk** ngay từ PARADIGM_SHIFT.md đầu tiên.
3. **Cite paper gốc**, không diễn giải lung tung.
4. **Empirical benchmark > formula citation**, luôn luôn.