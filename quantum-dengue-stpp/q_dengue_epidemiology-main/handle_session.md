# Session handoff — q_dengue_epidemiology

> Ghi lại trạng thái phiên làm việc để tiếp tục sau. Cập nhật timestamp mỗi lần quay lại.
> Last updated: 2026-07-23 (session 2)

## 0. Session 2 — đã làm

- **Môi trường mới** (sandbox reset): không có numpy/pandas/etc, đã `pip install -r requirements.txt` + `pip install pandas`
  (pandas thiếu trong requirements.txt gốc dù `bench_taiwan_kaohsiung.py` cần — **đã thêm vào requirements.txt**).
- **Đã fix `NEIGHBORHOOD_RADIUS_M`** trong `benchmarks/bench_taiwan_kaohsiung.py`: 1500m → **500m**.
  Đã đo thử nhiều radius trước khi chọn (largest component, arc_dim = N×mean_degree):
  ```
  radius=400   largest_N=279  mean_deg=5.50   arc_dim=1534   mem/matrix=18.8MB
  radius=500   largest_N=329  mean_deg=7.55   arc_dim=2484   mem/matrix=49.4MB   <- đã chọn
  radius=600   largest_N=351  mean_deg=10.14  arc_dim=3560   mem/matrix=101MB
  radius=1000  largest_N=522  mean_deg=20.38  arc_dim=10636  mem/matrix=905MB
  radius=1500  largest_N=641  mean_deg=34.03  arc_dim=21816  mem/matrix=3.8GB   <- bản cũ, treo máy
  ```
- **Đã chạy thành công** `benchmarks/bench_taiwan_kaohsiung.py` (radius=500m, N=329, mean_degree=7.55) — không crash, chạy nhanh (~vài chục giây).
  Kết quả lưu ở `output/taiwan_kaohsiung_benchmark.json`.

### Kết quả khoa học chính (trả lời câu hỏi ở mục 3 cũ)

**Kaohsiung (real, dense, đúng bán kính sinh học Aedes) VẪN KHÔNG có quantum resonance** — giống hệt kết luận âm tính của Điện Biên synthetic:
- Quantum peak P(marked) chỉ ~0.0001–0.0015 (target = top-5 hotspot thật), ngưỡng crossing 0.05 KHÔNG bao giờ đạt được → `crossing_t=None` cho mọi target thử.
- Classical hitting time thật lớn (3000–7300 bước) nhưng quantum không hề "bắt kịp" nhanh hơn.
- → Củng cố giả thuyết: đây không phải do graph Điện Biên "không thực tế", mà là đặc tính chung của graph dịch tễ đô thị/nông thôn (nhiều component nhỏ khi threshold theo khoảng cách sinh học, cấu trúc không đối xứng đủ để tạo resonance kiểu Szegedy). Kết quả âm tính có vẻ **robust across cả 2 loại địa hình thật**.
- Durr-Hoyer max-finding trên risk score thật: match đúng (idx=39, Dingxi Vil., 408 ca) — nhưng lưu ý **Bug 1 ở mục 1 vẫn chưa fix**, nên kết quả match này có thể "may mắn đúng" chứ không phải bằng chứng thuật toán đúng (xem mục 1).

### Đã fix (session 2, sau khi user chọn "Fix cả 2 bug ngay")

- [x] **Bug 1** (`src/durr_hoyer_max.py`, `grover_search()`): đo lường giờ dùng
  `rng.choice(n, p=marked_probs/total)` thay vì `argmax` tất định. `rng` được truyền từ
  `dur_hoyer_max_finding`/`dur_hoyer_benchmark` xuống `grover_search` (đổi signature, thêm tham số `rng`).
  Cũng bỏ cap `n_iter = min(n_iter, sqrt(n))` (không cần thiết, xem phân tích trong code).
  Cap `max_iter` ở outer loop (2 chỗ) đổi từ `ceil(sqrt(n))` → `n` (chỉ là safety bound, điều kiện dừng
  thật là "no improvement").
- [x] **Bug 2**: `M = sum(oracle.values > threshold)` giờ tính `oracle.count += n` (chi phí quét toàn bộ
  để biết marked-set), không đọc "miễn phí" nữa.
- [x] Đã chạy lại `python src/durr_hoyer_max.py`, `python benchmarks/bench_quantum_vs_classical.py`,
  `python benchmarks/bench_taiwan_kaohsiung.py` (tất cả với `PYTHONIOENCODING=utf-8`, xem mục dưới) —
  kết quả mới: **DH/Cl giờ > 1 (3.6–5.1×)** thay vì <1 trước đây (số liệu cũ là "tốt giả tạo" do bug).
  Match rate 67–100% (đúng — measurement giờ thật sự probabilistic).
  Kaohsiung: DH/classical = 10.10×, vẫn match đúng argmax thật (idx=39, Dingxi Vil.) ở lần chạy này.
- [x] **README.md đã cập nhật**: bảng Grover benchmark (số mới, giải thích DH/Cl>1 đúng lý thuyết),
  thêm dòng Kaohsiung vào bảng quantum walk, thêm câu tóm tắt "negative result robust trên cả 2 loại địa hình".
- [x] `requirements.txt`: đã thêm `pandas>=2.0` (thiếu, cần cho `bench_taiwan_kaohsiung.py`).

## 5. Session 2 (tiếp) — A/B/C: thử cải thiện quantum resonance trên Kaohsiung

User đồng ý thử lần lượt 3 hướng để xem có phục hồi được resonance không (xem lý thuyết ở mục 
`docs/QUANTUM_ADVANTAGE_REPORT.md`: cần connected + near-regular degree + low weight variance).

**A. k-NN regularization** (`benchmarks/bench_taiwan_knn.py`, mới tạo) — ép degree đều bằng k-NN
(binary và weighted), so với baseline radius=500m (peak_p=0.0001).
→ **KHÔNG đủ**: k=4 binary cho N=847 (1 component!) mean_deg=5.0, peak_p chỉ nhích lên **0.0032**
— vẫn cách ngưỡng 0.05 hơn chục lần. k=6+ bị skip vì arc_dim>6000 (an toàn RAM).
→ Kết luận: riêng việc làm degree đều không đủ để tạo resonance.

**B. Dense sub-region** (`benchmarks/bench_taiwan_dense.py`) — thu nhỏ về 1 quận dày nhất
(Sanmin Dist., 86 làng, chứa 4/5 top hotspot thật kể cả global max).
→ **Kết quả ban đầu trông rất ấn tượng**: peak_p 0.55–0.99, "speedup" tới 1307× — NHƯNG đây là
**ARTIFACT phương pháp luận**, không phải resonance thật.

**Đã phát hiện + fix bug phương pháp luận:** script (và cả `bench_taiwan_kaohsiung.py` gốc) luôn
dùng `start_v=0` cố định cho quantum walk, trong khi `classical_hitting_weighted()` lấy trung bình
hitting time từ **random start** mỗi trial → so sánh khập khiễng. Kiểm tra lại thì vertex 0 ở Sanmin
Dist. tình cờ là làng **gần thứ 2** với hotspot thật (412m/86 làng) — gần như đã kề cạnh target.

**Đã verify bằng `benchmarks/bench_taiwan_dense_v2.py`** (test nhiều start vertex khác nhau, cùng 1
graph radius=1000m, N=86, mean_deg=30.91):
```
start                   dist_to_marked(m)    t_class  crossing_t   peak_p  peak_t
closest-to-marked                   411.7    1307.43           1   0.4371       9   <- vertex 0 gốc
farthest-from-marked               4024.3    1307.43        None   0.0077    1904
random-0..4                    2021–2953    1307.43        None   0.003–0.014   ~1000-1800
```
→ **Chỉ start "may mắn" (gần kề target) mới resonance.** Với start xa hoặc random (đại diện thực
tế hơn — vì source case ban đầu không biết trước sẽ ở đâu), **vẫn KHÔNG resonance**, kết quả giống
hệt whole-city case. **B cũng KHÔNG khắc phục được** một khi sửa đúng phương pháp so sánh.

**C. Adapt coin/walk operator** — đã đọc lại `build_coined_walk()` trong `bench_weighted_walk.py`:
coin đã dùng **weighted Grover coin đúng chuẩn** (`psi = sqrt(A[v,u]/deg[v])`, phản xạ quanh phân
phối xác suất thật theo trọng số, KHÔNG phải naive uniform-Grover) — nên hướng "coin robust hơn với
weight heterogeneity" mà tôi đề xuất ban đầu **đã được implement sẵn**, không cần sửa thêm.
Hướng còn lại thật sự khác biệt về lý thuyết: implementation hiện tại là **coined quantum walk**
(arc-space, Grover-coin) — không phải **Szegedy bipartite walk** (discriminant matrix
D(x,y)=√(P(x,y)P(y,x))), vốn có **chứng minh toán học** quadratic speedup cho MỌI reversible
Markov chain (không cần degree đều) dựa trên spectral gap δ và tỉ lệ marked ε: hitting time
O(1/√(δε)). README có cite Wong 2015 "Equivalence of Szegedy's and coined quantum walks" nhưng
equivalence đó có điều kiện — trên graph bất định/lệch như Kaohsiung có thể KHÔNG tương đương.
→ Đây là việc lớn hơn hẳn (viết lại toàn bộ walk operator, không phải chỉnh tham số), **chưa làm**,
đang chờ user xác nhận có muốn đầu tư implement không trước khi làm.

### Việc còn lại (chưa làm)

- [ ] Lưu ý môi trường: chạy script trực tiếp bằng `python script.py` trên Windows console (cp1252) sẽ **crash `UnicodeEncodeError`** ở bất kỳ print nào có ký tự Unicode (→, ✓, ✗, ü, ê...) — rất nhiều file trong `src/` và `benchmarks/` có ký tự này. Workaround hiện tại: chạy với `PYTHONIOENCODING=utf-8 python script.py`. Chưa fix tận gốc (chưa hỏi user có muốn không — sẽ phải sửa nhiều file, hoặc thêm `sys.stdout.reconfigure(encoding="utf-8")` ở đầu mỗi entrypoint).
- [ ] (Optional, việc lớn) viết `src/graph_taiwan.py` tích hợp vào `pipeline.py` — vẫn chưa làm, càng ít động lực hơn vì kết quả Kaohsiung cũng âm tính (không phải hotspot để show off).
- [ ] Chưa commit các thay đổi session 2 vào git (working tree sạch lúc bắt đầu session — user cần xác nhận muốn commit gì trước khi làm).


## 1. Bug đã tìm thấy trong code hiện tại — CHƯA FIX (chỉ mới report cho user)

File: `src/durr_hoyer_max.py`, hàm `grover_search()` (dòng ~62-107) và `dur_hoyer_max_finding()` (dòng ~114-169).

**Bug 1 (nghiêm trọng):** Bước "đo lường" lượng tử dùng `np.argmax(marked_probs)` (dòng 106) — vì mọi marked-state có amplitude giống hệt nhau về mặt toán học, `argmax` luôn trả về **index nhỏ nhất** một cách tất định, không phải random-sample theo xác suất như phép đo lượng tử thật. Kết hợp với `max_iter = ceil(sqrt(n))` bị cắt sớm (dòng 142) → hàm có thể trả về **kết quả SAI (không phải argmax thật) mà không có cảnh báo gì**.

- Đã verify thực nghiệm: N=32, seed=42, risk thật từ Điện Biên → cap mặc định (6 iterations) trả về idx=22 SAI; chỉ cần thêm 2 iterations (cap=8) là ra đúng idx=30.
- Đây chính là nguyên nhân thật của "match rate 33%/67%" trong README — KHÔNG phải do "Grover search is probabilistic" như docstring/README đang giải thích. Diễn giải này sai.

**Bug 2 (phụ):** `M = np.sum(oracle.values > threshold)` (dòng 69) và `marked = oracle.values > threshold` (dòng 85) đọc thẳng mảng `oracle.values`, KHÔNG qua `oracle.query()` → oracle-query count đo được thấp hơn thực tế, làm tỉ lệ DH/Classical trong README (0.13–0.40×) tốt hơn giả tạo.

**Đề xuất fix (chưa làm):**
1. Thay `argmax` bằng `rng.choice(..., p=probs/probs.sum())` — random sample đúng bản chất phép đo.
2. Bỏ cap `max_iter` cứng, hoặc thêm điều kiện dừng dựa trên hội tụ thật (M==0).
3. Tính M kiểu BBHT "unknown M" thay vì đọc thẳng oracle.values, hoặc tính vào cost nếu vẫn giữ cách này.

→ **Việc cần làm:** hỏi lại user có muốn fix 2 bug này không (đã hỏi 1 lần, chưa có câu trả lời dứt khoát — bị chuyển hướng sang chủ đề dataset).

## 2. Dataset research — kết luận

- **Điện Biên synthetic (hiện tại):** KHÔNG phù hợp cho quantum walk resonance — 6 components, mean degree 2.86, weight variance 3500× (đã tự nhận trong `docs/QUANTUM_ADVANTAGE_REPORT.md`).
- **OpenDengue:** không dùng được — Việt Nam chỉ có ở admin-1 (tỉnh), dữ liệu dừng ở 2010, không có graph/mobility.
- **Recife, Brazil (Zenodo):** tốt (bairro-level, ~94 neighborhoods, real geocode) nhưng user muốn châu Á.
- **Đã chọn: Taiwan CDC Open Data** — xem mục 3 bên dưới, đang triển khai dở.
- Backup nếu Taiwan không ổn: Singapore NEA dengue clusters (GeoJSON, data.gov.sg) hoặc dataset published Huang et al. 2022 (Figshare, Scientific Data) — ~1.35km²/cluster, giai đoạn lockdown 2020.

## 3. Việc đang làm dở — Taiwan CDC Kaohsiung real-data pipeline

**Đã xong:**
- [x] Tải `data/taiwan_dengue_daily.csv` (31.4MB, 107,030 ca, 1998–2024/06/22). Lưu ý: HTTPS tới `od.cdc.gov.tw` bị timeout từ sandbox này — phải dùng `http://` (không phải `https://`) mới tải được, mất ~110s.
- [x] Cột dữ liệu quan trọng: `Village_Living`, `Village_Living_Code`, `Township_living`, `County_living`, `Enumeration_unit_lat/long` (toạ độ GPS thật per-case), `Number_of_confirmed_cases`.
- [x] Lọc + tổng hợp Kaohsiung City (vùng bùng dịch lớn nhất: đỉnh điểm 2014-2015, ~34,827 ca 2 năm đó) → `data/taiwan_kaohsiung_villages.csv`: **847 làng**, mỗi làng có `village_name, township, lat, lon, n_cases` (đã lọc outlier toạ độ ngoài bbox 22.4-23.3°N / 120.1-120.9°E).
- [x] Đo khoảng cách láng giềng gần nhất thật: median = **291m** — khớp rất sát với bán kính phát tán muỗi Aedes (~200m) mà chính graph_dien_bien.py giả định — TỐT HƠN HẲN Điện Biên synthetic về mặt sinh học.
- [x] Bảng mean-degree theo radius đã đo (dùng để chọn tham số):
  ```
  radius=500m:  mean_degree=3.75,  isolated=249/847
  radius=1000m: mean_degree=13.30, isolated=103/847
  radius=1500m: mean_degree=26.46, isolated=49/847
  radius=2000m: mean_degree=42.32, isolated=18/847
  radius=3000m: mean_degree=80.70, isolated=7/847
  radius=5000m: mean_degree=173.85, isolated=2/847
  ```
- [x] Viết `benchmarks/bench_taiwan_kaohsiung.py` — build graph thật (cùng kernel `exp(-d/200m)` như graph_dien_bien.py), restrict về largest component, chạy `quantum_search_run` (từ bench_weighted_walk.py) + `dur_hoyer_max_finding` (từ durr_hoyer_max.py) trên risk = log1p(n_cases) normalized.

**LỖI — chưa fix:**
- [ ] Chạy thử với `NEIGHBORHOOD_RADIUS_M = 1500.0` → mean degree ~26 trên 847 node → arc-space dimension ~20,800 chiều → ma trận dense trong `build_coined_walk()` (từ bench_weighted_walk.py) chiếm ~3.4GB mỗi ma trận (coin, shift, U_walk) → **treo máy, ăn hết RAM/CPU, phải kill process (PID 424796, đã kill -9)**.
- [ ] **VIỆC CẦN LÀM TIẾP THEO:** sửa `NEIGHBORHOOD_RADIUS_M` trong `benchmarks/bench_taiwan_kaohsiung.py` xuống ~500-800m (mean degree thấp hơn nhiều, ví dụ ~4-8) để arc-space dimension đủ nhỏ cho dense-matrix simulation chạy được trong thời gian hợp lý (so sánh: Điện Biên N=130, mean_degree=2.86 → arc-dim ~372, chạy nhanh; cần giữ scale tương tự, KHÔNG vượt quá vài nghìn chiều).
- [ ] Sau khi chạy được: so sánh kết quả (peak P(marked), resonance có xảy ra không) với kết quả âm tính của Điện Biên — đây là câu hỏi khoa học chính đang muốn trả lời: liệu graph đô thị thật (dày, khoảng cách nhỏ, khớp sinh học) có cho quantum resonance tốt hơn graph nông thôn synthetic hay không.
- [ ] Cân nhắc: nếu vẫn quá chậm ở radius nhỏ, có thể cần subset nhỏ hơn nữa (vd chỉ 1-2 township trong Kaohsiung, hoặc chỉ villages có ca trong giai đoạn đỉnh dịch 2014-2015) để giữ N nhỏ (~100-150, giống scale Điện Biên) thay vì cả 847 làng.
- [ ] (Optional, việc lớn hơn) Nếu kết quả khả quan, cân nhắc viết `src/graph_taiwan.py` chính thức theo interface giống `DienBienGraph`/`build_synthetic_dien_bien()` để cắm vào `pipeline.py` thay thế hẳn Điện Biên — hiện tại `bench_taiwan_kaohsiung.py` chỉ là script thử nghiệm độc lập, chưa tích hợp vào pipeline chính.

## 4. File map (những gì đã tạo trong session này)

- `data/taiwan_dengue_daily.csv` — raw, 107k rows, KHÔNG commit vào git nếu có git (file lớn).
- `data/taiwan_kaohsiung_villages.csv` — 847 villages, đã aggregate, sẵn sàng dùng.
- `benchmarks/bench_taiwan_kaohsiung.py` — draft benchmark, **cần sửa radius trước khi chạy lại**.
