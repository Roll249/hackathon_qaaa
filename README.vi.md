# Q-STPP — Tăng cường dữ liệu "lấy cảm hứng từ lượng tử" cho Spatio-Temporal Point Process của sốt xuất huyết

**Dự án hackathon (QC4SG).** Repo này nghiên cứu **hoán vị Second-Order Preserving (SOP)**
— một kỹ thuật tăng cường dữ liệu (data augmentation) cho các quá trình điểm không gian–thời
gian (Spatio-Temporal Point Process, STPP) (Mohler & Mateu, 2024) — và so sánh ba chiến lược
tìm kiếm cục bộ (local search) cổ điển để sinh ra các hoán vị đó: một là bộ lấy mẫu
Metropolis-Hastings thuần túy, hai còn lại mượn *ý tưởng* từ các thuật toán lượng tử (Grover,
QAOA) nhưng **không chạy bất kỳ mạch lượng tử nào**. Repo cũng chứa một bộ dữ liệu giám sát
sốt xuất huyết thật của Đông Nam Á (dẫn xuất từ OpenDengue) mà pipeline có thể chạy trực tiếp
lên.

> **Phạm vi trung thực — đọc phần này trước.** Không có phần cứng lượng tử và không có
> simulator lượng tử ở bất kỳ đâu trong repo này. Mọi con số trong mọi file kết quả đều được
> sinh ra bởi code cổ điển NumPy/SciPy/pandas. "Grover-inspired" và "QAOA-inspired" chỉ là tên
> gọi cho các heuristic cổ điển mượn *ý tưởng* từ hai thuật toán đó (khuếch đại tập trung,
> nhiễu loạn kiểu mixer) — không có gì ở đây chứng minh một lợi thế lượng tử (quantum
> advantage). Các phiên bản trước của dự án (v9–v12) từng đưa ra những tuyên bố lớn hơn, sau
> đó được phát hiện là không hợp lệ và đã bị rút lại chính thức; xem
> [`quantum-dengue-stpp/DEVELOPMENT_HISTORY.md`](quantum-dengue-stpp/DEVELOPMENT_HISTORY.md).
> Tài liệu này, và phần code nó mô tả, chủ đích chỉ nói về những gì thực sự đã chạy và đã đo
> được.

---

## 1. Cấu trúc repo

```
hackathon_qaaa/
├── dengue_dataset/            # dữ liệu dengue thật (OpenDengue-derived) + script chuẩn bị dữ liệu
├── quantum-dengue-stpp/       # pipeline Q-STPP (bản synthetic + bản dữ liệu thật)
├── output_dataset/            # kết quả chạy trên 8 quốc gia với dữ liệu thật (xem mục 7)
├── message.txt                # đề xuất hướng nghiên cứu (tầm nhìn tương lai, chưa hiện thực)
├── improve.md                 # ghi chú khám phá về các hướng QML khả dĩ (chưa hiện thực)
└── S7-ECSIA-2025-Prague.pdf   # bài báo tham chiếu (Mateu, S7-ECSIA 2025)
```

### 1.1 `dengue_dataset/` — dữ liệu thật

Dữ liệu giám sát sốt xuất huyết Đông Nam Á dẫn xuất từ
[OpenDengue](https://opendengue.org), bao phủ 11 quốc gia SEA (độ phân giải năm) và 8 quốc
gia / 233 vùng admin1 (độ phân giải tháng).

| File | Nội dung |
|---|---|
| `sea_dengue_spatial.csv` | Số ca theo năm, nhiều độ phân giải không gian, 11 quốc gia |
| `sea_dengue_admin1_month.csv` | Số ca theo tháng ở cấp admin1 (tỉnh/bang), 8 quốc gia, 1993–2022 |
| `sea_dengue_admin1_month_pivot.csv` | Cùng dữ liệu trên, pivot theo ngày × vùng |
| `filter_sea.py` | Lọc bộ trích xuất OpenDengue gốc xuống 11 quốc gia SEA |
| `make_training_set.py` | Lọc còn `S_res=Admin1, T_res=Month`, sắp xếp, làm sạch |
| `make_pivot.py` | Dựng bảng pivot ở trên |
| `eda_analysis.py` / `eda_analysis_v2.py` | Phân tích khám phá dữ liệu (thống kê theo quốc gia, tính mùa vụ, phát hiện năm dịch, vẽ hình) |
| `check_training_set.py`, `inspect_data.py` | Script kiểm tra nhanh chất lượng dữ liệu |

Không file nào trong số này liên quan đến lượng tử — đây thuần túy là chuẩn bị dữ liệu bằng
pandas.

### 1.2 `quantum-dengue-stpp/` — pipeline chính

Thí nghiệm thực sự. Hai điểm chạy (entry point) dùng chung phần lõi code:

| File | Nguồn dữ liệu | Mục đích |
|---|---|---|
| `run_q_stpp_v15_fair.py` | Mô phỏng Hawkes-process tổng hợp (synthetic) | Benchmark gốc, được kiểm soát hoàn toàn |
| `run_q_stpp_v15_real.py` | Dữ liệu thật `dengue_dataset/sea_dengue_admin1_month.csv` | Cùng benchmark, dùng ngày sự kiện + vùng thật (mục 4) |

Tài liệu bổ trợ: `ARCHITECTURE.md` (thiết kế hệ thống), `THEORY.md` (nền tảng L-function +
local search), `Q_STPP_V15_REPORT.md` (mẫu báo cáo phương pháp luận), `DEVELOPMENT_HISTORY.md`
(lịch sử đầy đủ, bao gồm các tuyên bố đã bị rút ở phiên bản trước), `archive/` (code v4–v12 đã
bị thay thế, giữ lại để lưu vết, không được dùng bởi bất cứ thứ gì hiện tại).

### 1.3 `output_dataset/` — kết quả chạy trên dữ liệu thật

Kết quả chạy `run_q_stpp_v15_real.py` trên cả 8 quốc gia có trong `dengue_dataset/`. Xem phân
tích đầy đủ ở mục 7.

---

## 2. Bài toán: hoán vị Second-Order Preserving là gì, và tại sao cần sinh ra chúng?

Một quá trình điểm không gian–thời gian là một tập các sự kiện, mỗi sự kiện có vị trí `(x, y)`
và thời điểm `t` — một ca báo cáo sốt xuất huyết là ví dụ tự nhiên. Các mô hình thống kê cho
loại quá trình này (và các mô hình deep learning ngày càng được dùng để dự báo chúng) cần
nhiều dữ liệu huấn luyện hơn số năm giám sát dịch tễ thực tế có thể cung cấp.

**Tăng cường SOP** (Mohler & Mateu, 2024) tạo ra tập sự kiện *mới*, tổng hợp, từ một tập
*thật* bằng cách **hoán đổi xem mốc thời gian nào gắn với vị trí nào**, trong khi cố gắng bảo
toàn cấu trúc không gian–thời gian bậc hai của mẫu — được tóm tắt bởi **hàm K/L của Ripley**,
đo mức độ tụ cụm hay phân tán của các sự kiện theo các khoảng cách phân tách `r`. Một hoán vị
giữ `L(r)` gần với bản gốc là "bảo toàn bậc hai" (second-order preserving): nó trông giống về
mặt thống kê với dữ liệu thật mà không phải là một bản sao của nó.

Một bộ sinh tăng cường hữu ích phải làm **đồng thời hai việc**:

1. **Bảo toàn `L(r)`** — sai số thấp so với `L(r)` của mẫu gốc.
2. **Giữ tính đa dạng** — sinh ra các hoán vị *khác nhau*, không phải lặp đi lặp lại cùng một
   (hoặc gần giống một) đáp án. Một phương pháp luôn hội tụ về cùng một đáp án sai số thấp thì
   vô dụng cho việc tăng cường dữ liệu: nó chỉ cho bạn một mẫu mới, không phải nhiều mẫu.

Câu hỏi cốt lõi của repo này là: **trong ba chiến lược tìm kiếm cổ điển, chiến lược nào đạt
được đánh đổi tốt nhất giữa hai mục tiêu trên, dưới cùng một ngân sách tính toán?**

---

## 3. Kiến trúc lõi — `run_q_stpp_v15_fair.py`, giải thích từng hàm

Đây là file mà mọi thứ khác import từ đó. Đọc từ trên xuống, đây chính là toàn bộ pipeline.

### 3.1 `simulate_hawkes(n_events_target, mu, theta, omega, T, space_size, seed)`

Sinh một mẫu điểm không gian–thời gian tự kích thích ("Hawkes-like") tổng hợp bằng thuật toán
thinning của Ogata: đề xuất các thời điểm sự kiện ứng viên từ một quá trình đồng nhất với
cường độ `lam_max`, chấp nhận mỗi ứng viên với xác suất `lam(t) / lam_max`, trong đó `lam(t) =
mu + Σ θ·ω·exp(-ω·dt)` cộng dồn ảnh hưởng suy giảm của các sự kiện quá khứ trong cửa sổ 2 đơn
vị thời gian (`dt < 2.0`). `lam_max` được tính lại ở mỗi bước dựa trên `n_active` — số sự kiện
còn nằm trong cửa sổ suy giảm đó — nên nó luôn là một cận trên *chặt*, thay vì trôi dần lên
theo tổng số sự kiện lịch sử (một lỗi đã được sửa trong phiên bản này; xem mục 8). Trả về
`(times, x, y)` dưới dạng ba mảng NumPy. Seed giúp toàn bộ mẫu tái lập chính xác 100%.

### 3.2 `compute_L_summary(times, coords_x, coords_y, r_values, T=1.0, space_size=1.0)`

Thống kê tóm tắt bậc hai mà mọi thứ khác đều xoay quanh.

1. Xếp chồng `(x, y, time·scale)` thành một đám mây điểm 3D (`time_scale = space_size / T`),
   coi thời gian như một trục "giả không gian" thứ ba.
2. Tính ma trận khoảng cách Euclid đầy đủ giữa mọi cặp điểm (`scipy.spatial.distance.pdist` +
   `squareform`).
3. Với mỗi bán kính `r` trong `r_values`, `K(r) = (số cặp có khoảng cách < r, trừ đi N cặp
   "tự thân" ở khoảng cách 0) / N²` — tỉ lệ cặp điểm là "hàng xóm" trong bán kính `r`, một
   phiên bản không gian–thời gian trực tiếp của K-function Ripley.
4. Trả về `L(r) = sign(K)·|K|^(1/3)` — một phép biến đổi căn bậc ba để ổn định phương sai.
   Đây **không** phải L-function 2D cổ điển của Ripley (`L(r) = √(K(r)/π) − r`); phép biến đổi
   chính xác không quan trọng cho việc so sánh này vì nó được áp dụng giống hệt nhau cho mọi
   phương pháp được so sánh, nên khác biệt *tương đối* vẫn có giá trị dù con số thô không phải
   L-function sách giáo khoa.

`r_values` mặc định là `np.linspace(0.05, 0.3, 8)` — 8 bán kính cách đều nhau — ở cả hai
pipeline.

### 3.3 `l_error(L_perm, L_target)`

Sai số bình phương trung bình (MSE) giữa đường `L(r)` của một hoán vị ứng viên và đường `L(r)`
của mẫu gốc, tính trung bình trên 8 bán kính. Đây là con số duy nhất mà mọi phương pháp tìm
kiếm cố gắng tối thiểu hóa.

### 3.4 `set_diversity(perms)`

Với một *tập* gồm `m` hoán vị, tính khoảng cách Hamming chuẩn hóa trung bình theo từng cặp
(mean pairwise normalized Hamming distance): với mỗi cặp hoán vị, tỉ lệ vị trí mà chúng khác
nhau, trung bình trên toàn bộ `m(m−1)/2` cặp. `0.0` nghĩa là mọi hoán vị trong tập giống hệt
nhau (mode collapse — vô dụng cho tăng cường dữ liệu); `1.0` nghĩa là mọi cặp khác nhau ở mọi
vị trí (đa dạng tối đa). Đây là chỉ số bắt được trường hợp một phương pháp tìm kiếm "gian lận"
bằng cách luôn trả về cùng một đáp án sai số thấp.

### 3.5 `_generate_perms(strategy, ..., n_perms, evals_per_perm, rng)` — ba phương pháp

Với mỗi hoán vị trong số `n_perms` hoán vị cần sinh, hàm bắt đầu từ một hoán vị ngẫu nhiên và
thực hiện `evals_per_perm − 1` bước tìm kiếm cục bộ, mỗi bước tốn *đúng một* lần gọi
`compute_L_summary` (chi phí chủ đạo `O(N²)`) — nên **mọi phương pháp tốn đúng một số lần đánh
giá L-summary như nhau**, đây chính là điều làm cho việc so sánh "công bằng". Ba chiến lược chỉ
khác nhau ở cách đề xuất ứng viên và cách chấp nhận nó:

| Chiến lược | Đề xuất | Chấp nhận |
|---|---|---|
| `mh` | tráo đổi 2 vị trí ngẫu nhiên | **Metropolis**: luôn chấp nhận nếu tốt hơn; chấp nhận ứng viên tệ hơn với xác suất `exp(-(Δerror)/temperature)`. Nhiệt độ (temperature) giảm dần theo cấp số nhân, từ mức sai số ban đầu xuống còn 1% của nó trong suốt lượt chạy — thay thế cho một nhiệt độ cố định, không chuẩn hóa ở phiên bản trước, vốn khiến MH chấp nhận ~90% các bước đi tệ hơn bất kể quy mô bài toán. |
| `grover` ("Grover-inspired") | tráo đổi 2 vị trí ngẫu nhiên | **Greedy (tham lam)**: chỉ chấp nhận nếu sai số của ứng viên thấp hơn nghiêm ngặt. Có nét tương đồng lỏng lẻo với khuếch đại biên độ (amplitude amplification) của Grover — luôn tiến về trạng thái "được đánh dấu" (tốt hơn) — nhưng được cài đặt như hill-climbing thông thường, không có biên độ, không có mạch lượng tử. |
| `qaoa` ("QAOA-inspired") | tráo đổi đồng thời 1 đến `~N/4` cặp vị trí ngẫu nhiên | **Greedy**, cùng quy tắc như `grover`. Đề xuất đa-tráo-đổi (multi-swap) có nét tương đồng lỏng lẻo với một mixer của QAOA làm nhiễu loạn nhiều phần tử cùng lúc — vẫn hoàn toàn cổ điển. |

### 3.6 `evaluate_method(strategy, ..., seed)`

Chạy `_generate_perms` cho một phương pháp (với `np.random.default_rng(seed)` khởi tạo mới —
*cùng một* giá trị seed được dùng lại cho cả ba phương pháp để cả ba xuất phát từ cùng một
hoán vị ngẫu nhiên ban đầu; chúng chỉ phân kỳ vì logic đề xuất/chấp nhận của mỗi phương pháp
tiêu thụ số ngẫu nhiên khác nhau, không bao giờ vì seed khác nhau) và trả về:

- `mean_error`, `std_error` — trung bình và độ phân tán của `l_error` trên `n_perms` hoán vị
  đã sinh (tính lại một lần nữa sau khi sinh xong, chỉ để phục vụ báo cáo).
- `diversity` — `set_diversity` của cùng `n_perms` hoán vị đó.
- `time` — thời gian thực tế (giây) chỉ cho quá trình sinh (phần tính lại để báo cáo ở trên
  không được tính vào đây).

### 3.7 `run_single`, `aggregate`, `print_summary`, `plot_summary`

- `run_single(seed, n_events, n_perms, evals_per_perm)`: dựng một mẫu điểm (từ
  `simulate_hawkes`, hoặc trong script dữ liệu thật, từ sự kiện thật), tính `L(r)` mục tiêu của
  nó, và chạy cả ba phương pháp trên mẫu đó. Trả về `None` (và nơi gọi in ra dòng `SKIPPED`)
  nếu mẫu có ít hơn 10 sự kiện.
- `aggregate(rows)`: tính trung bình `mean_error`, `diversity`, `time` trên mọi seed có cùng
  N mục tiêu.
- `print_summary(agg)`: bảng in ra console (N × phương pháp → sai số, đa dạng, thời gian).
- `plot_summary(agg, path)`: file PNG 2 panel — L(r) error theo N, và đa dạng theo N, mỗi
  phương pháp một đường.

### 3.8 Vòng quét (`main`)

Lặp qua mọi tổ hợp `(N, seed)` được yêu cầu trên dòng lệnh, gọi `run_single` cho từng tổ hợp,
tổng hợp, in ra, và ghi `results.json` + một file plot PNG. Mặc định: `seeds = [1..5]`,
`n_events = [20, 30, 50]`, `n_perms = 10`, `evals_per_perm = 200`.

---

## 4. Bản dữ liệu thật — `run_q_stpp_v15_real.py`

Import nguyên vẹn mọi hàm ở trên (`compute_L_summary`, `evaluate_method`, `aggregate`,
`print_summary`, `plot_summary`) — chỉ có phần **nạp dữ liệu (data loader)** là mới, nên thuật
toán tìm kiếm và giao thức công bằng giống hệt mục 3.

### 4.1 `load_real_events(df, country, year_start, year_end, seed, jitter_deg, max_events)`

Dựng một mẫu điểm `(times, x, y)` từ `sea_dengue_admin1_month.csv`:

1. Lọc theo `country` và khoảng năm yêu cầu, chỉ giữ các dòng có `dengue_total > 0` — **mỗi
   sự kiện tương ứng với một bản ghi (vùng admin1, tháng)**, không phải một sự kiện cho mỗi ca
   bệnh. (Biến mỗi ca bệnh riêng lẻ thành một điểm riêng sẽ tạo ra hàng triệu điểm mà dữ liệu
   gốc vốn cũng không có tọa độ theo từng ca — xem mục 4.2 để biết chính xác cái gì là thật, cái
   gì là proxy.)
2. Nếu số dòng khả dụng nhiều hơn `max_events`, lấy mẫu ngẫu nhiên xuống còn `max_events`
   (có seed, nên tái lập được).
3. **Thời gian**: ngày trong tháng của mỗi sự kiện được lấy ngẫu nhiên đều (dữ liệu gốc chỉ có
   độ phân giải tháng, không có độ phân giải ngày). Số ngày trôi qua kể từ sự kiện sớm nhất
   trong mẫu được lấy sau đó co giãn tuyến tính vào khoảng `[0, 2·jitter_deg]` — cùng thang số
   với độ jitter không gian (mục 4.2) — để thời gian và không gian đóng góp tương đương vào
   metric khoảng cách bên trong `compute_L_summary`.
4. **Không gian**: `_region_xy(region, ...)` băm (hash) tên vùng bằng SHA-256 thành một seed,
   rồi rút ra một độ lệch cố định `(±jitter_deg, ±jitter_deg)` quanh centroid của quốc gia. Cùng
   một vùng → cùng một độ lệch, ở mọi lượt chạy, mãi mãi — nên các điểm thuộc cùng một vùng tụ
   lại với nhau và các điểm thuộc vùng khác tách ra, mà không phụ thuộc vào thứ tự lặp hay việc
   lấy mẫu con `max_events`.

### 4.2 Cái gì là thật, cái gì là proxy — đọc trước khi diễn giải bất kỳ con số nào

| Khía cạnh | Trạng thái |
|---|---|
| Vùng (admin1, tháng) nào có ca bệnh | **Thật** — từ OpenDengue |
| Tháng của mỗi sự kiện | **Thật** |
| Ngày trong tháng | Ngẫu nhiên hóa (dữ liệu gốc không có độ phân giải mịn hơn) |
| Khoảng cách không gian giữa các vùng | **Proxy** — độ jitter cố định theo từng vùng quanh centroid quốc gia; dataset này không có dữ liệu centroid/ranh giới cấp admin1 |
| Kinh độ/vĩ độ tuyệt đối của bất kỳ sự kiện nào | **Proxy** — không nên coi là địa lý thật |
| Số ca bệnh (`dengue_total`) | Chỉ dùng làm điều kiện lọc `> 0`; **không** dùng để gia trọng hay nhân bản sự kiện |

Vì tọa độ không gian là proxy, giá trị **tuyệt đối** của sai số `L(r)` không có ý nghĩa vật lý.
Điều **có** ý nghĩa: so sánh **tương đối** giữa `mh`, `grover`, và `qaoa` trên *cùng* một tập
điểm — đúng lưu ý đã áp dụng cho pipeline synthetic (mục 3.2), chỉ thêm một lý do thứ hai đặc
thù cho bộ dữ liệu này.

### 4.3 Dòng lệnh

```bash
python3 run_q_stpp_v15_real.py \
  --country CAMBODIA \
  --year_start 2005 --year_end 2006 \     # bỏ cả hai để tự động dùng toàn bộ record của quốc gia
  --seeds 1 2 3 4 5 \
  --max_events 30 60 120 \
  --n_perms 10 --evals_per_perm 200 \
  --out_dir output_dataset/cambodia        # bỏ để dùng output_result/q_stpp_v15_real/
```

Nếu bỏ `--year_start`/`--year_end`, script tự động dò năm đầu và năm cuối có
`dengue_total > 0` cho quốc gia đó, nên mặc định mỗi quốc gia dùng toàn bộ record thật khả dụng
của mình.

---

## 5. Cách chạy pipeline

```bash
cd quantum-dengue-stpp
pip install -r requirements.txt          # numpy, scipy, matplotlib, pandas

# Benchmark Hawkes tổng hợp (synthetic)
python3 run_q_stpp_v15_fair.py
# hoặc: ./run.sh          (./run.sh smoke để kiểm tra nhanh)

# Benchmark dữ liệu thật, một quốc gia
python3 run_q_stpp_v15_real.py --country "VIET NAM"
```

Cả hai đều ghi ra một file `*_results.json` (số liệu thô + tổng hợp) và một `*_plot.png` (sai
số và đa dạng theo N) vào thư mục output tương ứng. Thời gian chạy từ vài giây đến vài phút cho
một lượt quét đầy đủ trên CPU laptop thường — không cần GPU, không cần quantum backend, không
cần cấp phát gì thêm.

---

## 6. Phạm vi trung thực — dự án này là gì, và không phải là gì

**Là:**
- Một phép so sánh có kiểm soát, cùng ngân sách tính toán, giữa ba heuristic tìm kiếm hoán vị
  *cổ điển* cho tăng cường dữ liệu SOP, báo cáo cả chất lượng (`L(r)` error) lẫn đa dạng, trên
  cả dữ liệu tổng hợp lẫn dữ liệu sự kiện thật.

**Không phải:**
- Một phép tính lượng tử dưới bất kỳ hình thức nào — không mạch, không simulator, không quantum
  backend ở bất cứ đâu trong phần code thực sự chạy.
- Một hệ thống dự báo dengue hay tiên đoán bùng dịch đã được kiểm chứng — không có gì ở đây
  huấn luyện hay đánh giá một mô hình dự báo; pipeline chỉ so sánh mức độ các phương pháp tìm
  kiếm hoán vị bảo toàn một thống kê tóm tắt tốt đến đâu.
- Có thể áp dụng cho địa lý admin1 thật khi tính khoảng cách không gian — xem mục 4.2.

---

## 7. Kết quả — chạy trên 8 quốc gia với dữ liệu thật (`output_dataset/`)

`output_dataset/` được sinh ra bằng cách chạy `run_q_stpp_v15_real.py` một lần cho mỗi quốc
gia SEA có trong dataset, mỗi lần dùng **toàn bộ record thật khả dụng** (khoảng năm tự động dò
ra), `seeds = [1..5]`, `max_events = [30, 60, 120]`. Cấu trúc:

```
output_dataset/
├── SUMMARY.md                          # bảng đầy đủ theo từng quốc gia — đọc file này để lấy số liệu chính xác
├── cambodia/      real_comparison_results.json, real_comparison_plot.png, run_log.txt
├── indonesia/     ...
├── laos/          ...
├── malaysia/      ...
├── singapore/     ...
├── thailand/      ...
├── timor_leste/   ...
└── vietnam/       ...
```

### 7.1 Mức độ sẵn có của dữ liệu theo quốc gia

| Quốc gia | Khoảng năm dùng | Số sự kiện (vùng, tháng) thật khả dụng |
|---|---|---|
| Cambodia | 1998–2010 | 2.713 |
| Indonesia | 2004–2006 | 394 |
| Lào (Lao PDR) | 1998–2010 | 1.335 |
| Malaysia | 1993–2010 | 1.903 |
| Singapore | 1993–2010 | 216 |
| Thái Lan | 1993–2022 | 31.685 |
| Timor-Leste | chỉ 2005 | 30 |
| Việt Nam | 1994–2010 | 7.483 |

### 7.2 Ý nghĩa từng cột trong `SUMMARY.md`

- **N (target)**: mức trần `--max_events` được yêu cầu cho dòng đó (30, 60, hoặc 120).
- **N (actual)**: số sự kiện thật thực sự được đưa vào lượt chạy đó. Bằng đúng N (target) khi
  đủ dữ liệu thật; bị giới hạn theo tổng lượng khả dụng của quốc gia nếu không đủ (xem
  Timor-Leste bên dưới).
- **Phương pháp**: `MH` (Metropolis-Hastings), `Grover-inspired`, `QAOA-inspired` — xem mục
  3.5.
- **L(r) err**: sai số bình phương trung bình của `L(r)` các hoán vị được sinh ra so với `L(r)`
  của mẫu thật, trung bình trên `n_perms=10` hoán vị và 5 seed. Thấp hơn là tốt hơn (bảo toàn
  cấu trúc không gian–thời gian gốc chặt chẽ hơn). Nhớ mục 4.2: giá trị *tuyệt đối* phụ thuộc
  vào tọa độ proxy; nên so sánh *giữa các phương pháp*, không so sánh *giữa các quốc gia*.
- **diversity**: khoảng cách Hamming chuẩn hóa trung bình theo từng cặp trong 10 hoán vị được
  sinh ra, trung bình trên 5 seed. Cao hơn nghĩa là phương pháp tạo ra các mẫu tăng cường thực
  sự khác nhau hơn, thay vì gần như bản sao của nhau.

### 7.3 Các quy luật quan sát được

Trên sáu quốc gia có đủ dữ liệu để đạt cả ba mức N (Cambodia, Indonesia, Lào, Malaysia, Thái
Lan, Việt Nam), ba quy luật nhất quán xuất hiện — mỗi quy luật kèm một lời giải thích cụ thể,
có thể kiểm chứng, thay vì chỉ diễn giải cảm tính:

**(a) `qaoa` và `grover` đạt sai số `L(r)` thấp hơn `mh`, gần như ở mọi nơi.** Cả hai đều là bộ
leo đồi tham lam thuần túy (mục 3.5: chỉ chấp nhận cải thiện nghiêm ngặt), trong khi `mh` dành
một phần ngân sách 200 lần đánh giá cố định của nó để cố ý chấp nhận các ứng viên tệ hơn (nhằm
tránh bị kẹt ở cực tiểu cục bộ). Dưới một ngân sách đánh giá *cố định*, tìm kiếm tham lam luôn
đạt sai số cuối cùng thấp hơn so với một bộ lấy mẫu có ủ nhiệt (annealed sampler) — đây là đánh
đổi kỳ vọng, đúng sách giáo khoa, không phải điều bất ngờ.

**(b) Đa dạng cao (0,96–0,99) ở cả *ba* phương pháp, kể cả hai phương pháp tham lam — một đánh
đổi chất lượng/đa dạng nhỏ hơn so với trực giác "tham lam = mode collapse" thường gợi ý.** Lý do
nằm ở ngân sách, không phải ở phương pháp: với `evals_per_perm=200` cố định bất kể `N`, một
lần tráo đổi chỉ làm nhiễu loạn `2/N` của hoán vị, nên khi `N` tăng vượt quá vài chục, 200 bước
leo đồi tham lam không thể hội tụ hoàn toàn về cực tiểu toàn cục — nó hội tụ về *một* cực tiểu
cục bộ nào đó, và cực tiểu cục bộ nào phụ thuộc rất nhiều vào hoán vị khởi tạo ngẫu nhiên và
chuỗi tráo đổi cụ thể. Kết quả là một tập vẫn đa dạng ngay cả dưới tìm kiếm tham lam, ở tỉ lệ
ngân sách/N này. (Một `evals_per_perm` lớn hơn nhiều được kỳ vọng sẽ làm giảm độ đa dạng của
hai phương pháp tham lam — đây là một hướng kiểm chứng khả thi, chưa được chạy ở đây.)

**(c) Đa dạng leo dần về trần thống kê của nó khi N tăng: ≈0,967 tại N=30, ≈0,983 tại N=60,
≈0,992 tại N=120 — khớp gần như chính xác với `(N−1)/N`** (29/30 = 0,9667; 59/60 = 0,9833;
119/120 = 0,9917). Đây chính là tỉ lệ vị trí kỳ vọng mà **hai hoán vị ngẫu nhiên độc lập** có độ
dài N khác nhau — trần toán học của chính chỉ số khoảng-cách-Hamming-chuẩn-hóa (một vị trí cố
định giữ cùng một giá trị ở hai hoán vị ngẫu nhiên đều độc lập với xác suất đúng bằng `1/N`).
Sự khớp gần như tuyệt đối này ở mọi quốc gia và mọi phương pháp cho thấy các tập hoán vị được
sinh ra đang hoạt động gần như các phép rút mẫu ngẫu nhiên *độc lập* ở mức ngân sách này, vì lý
do đã nêu ở (b) — đây là tính chất của **chỉ số đo và tỉ lệ ngân sách/N**, không phải bằng chứng
rằng bất kỳ phương pháp nào "sáng tạo hơn" khi N tăng.

**(d) Sai số `L(r)` cũng thu nhỏ khi N tăng** (ví dụ Cambodia, MH: 0,000187 → 0,000039 →
0,000005 khi N=30→60→120). Một phần là do tìm kiếm hội tụ tốt hơn theo (a)/(b); một phần là do
`K(r) = pairs/N²` tự nó là một ước lượng có phương sai thấp hơn khi N lớn hơn (nhiều cặp hơn để
trung bình hóa nhiễu lấy mẫu), nên ngay cả một hoán vị khá ngẫu nhiên cũng có xu hướng có `L(r)`
gần với mục tiêu hơn khi N tăng. Báo cáo này không phân tách được bao nhiêu phần của sự sụt
giảm đến từ "tìm kiếm tốt hơn" so với "chỉ số đo trở nên mượt hơn" — nêu đây là câu hỏi mở, chứ
không khẳng định chất lượng tìm kiếm cải thiện theo N.

### 7.4 Lưu ý riêng theo từng quốc gia — đọc trước khi trích dẫn bất kỳ con số đơn lẻ nào

- **Singapore là một trường hợp suy biến toán học, không phải một phát hiện.** Dataset này chỉ
  có **một** vùng admin1 cho Singapore, nên mọi sự kiện dùng chung đúng một tọa độ không gian
  proxy. Với độ biến thiên không gian bằng 0, *mọi* hoán vị mốc thời gian đều tạo ra đúng cùng
  một *tập* điểm (chỉ có sự tương ứng thời-gian-với-chỉ-số thay đổi, và — vì mọi điểm dùng
  chung một `(x,y)` — sự tương ứng đó vô hình với một thống kê tóm tắt tính trên đám mây điểm
  không có thứ tự). Do đó sai số `L(r)` bằng đúng `0,000000` cho cả ba phương pháp ở cả ba mức
  N: đây là hệ quả trực tiếp của việc Singapore chỉ có một vùng, không phải tuyên bố rằng bất kỳ
  phương pháp nào đã "giải quyết" được bài toán.
- **Các dòng N=60 và N=120 của Timor-Leste không phải mẫu lớn hơn.** Chỉ có 30 sự kiện (vùng,
  tháng) thật tồn tại trong toàn bộ record khả dụng (2005 là năm duy nhất có dữ liệu). Các lượt
  chạy `max_events=60` và `max_events=120` do đó rơi về lại đúng 30 sự kiện giống như
  `max_events=30` (mục 4.1, bước 2 chỉ lấy mẫu con *xuống*, không bao giờ độn *lên*) — ba dòng N
  của Timor-Leste trong `SUMMARY.md` vì vậy giống hệt nhau.

### 7.5 Kết quả này *không* chứng minh điều gì

- **Không** phải kết quả lượng tử — mọi con số ở trên đều đến từ code cổ điển NumPy/pandas
  (mục 6).
- **Không** phải một benchmark dự báo bùng dịch đã được kiểm chứng — kết quả này đo chất lượng
  tìm kiếm hoán vị so với một thống kê tóm tắt, không đo độ chính xác dự báo so với số ca thật
  bị giữ lại để kiểm định (held-out).
- **Không** phải bằng chứng về mức độ tụ cụm không gian thật của dengue ở cấp admin1 — tọa độ
  không gian đứng sau các con số này là proxy (mục 4.2); một phân tích địa lý thật sự cần
  centroid/ranh giới admin1 thật, thứ mà dataset này không cung cấp.

---

## 8. Các lỗi đã sửa trong quá trình rà soát này (để lưu vết)

- Cận trên `lam_max` của `simulate_hawkes` giờ tỉ lệ theo `n_active` (số sự kiện còn nằm trong
  cửa sổ suy giảm 2 đơn vị thời gian) thay vì tổng số sự kiện lịch sử — trước đây điều này khiến
  xác suất chấp nhận sự kiện mới tụt dần khi lượt chạy kéo dài.
- Console output giờ ép buộc dùng UTF-8 (`sys.stdout.reconfigure`) để các ký tự vẽ khung
  (box-drawing) trong bảng tổng hợp không làm crash chương trình trên codepage mặc định `cp1252`
  của Windows.
- Một dạng lỗi âm thầm đã được sửa: khi một cell `(seed, N)` sinh ra quá ít sự kiện để chạy
  (dưới 10), trước đây nó biến mất khỏi kết quả mà không để lại dấu vết. Giờ nó in ra một dòng
  `SKIPPED` rõ ràng.
- Đã xóa hàm `ratio()` không dùng đến (dead code, chưa từng được gọi ở đâu), cùng với câu tuyên
  bố mô tả nó trong báo cáo.

---

## 9. Hướng phát triển tương lai (chưa hiện thực — xem `message.txt`)

`message.txt` (tiếng Việt) phác thảo một hướng nghiên cứu dài hơi hơn: thay thế SOP cổ điển
bằng các **mô hình sinh lượng tử** thực sự (QGAN / Quantum Born Machine / VQC generator) huấn
luyện trên dữ liệu dengue thật, và so sánh độ chính xác dự báo (RMSE, MAE, F1) có và không có
tăng cường lượng tử. Không có gì trong số đó được hiện thực trong repo này — code hiện tại
chính xác là những gì được mô tả ở mục 3–7, không hơn.

## Tài liệu tham khảo

- Mateu, J. (2025). *Statistical learning for spatio-temporal point processes*
  (S7-ECSIA, Prague) — `S7-ECSIA-2025-Prague.pdf`.
- Mohler, G. & Mateu, J. (2024). Second-order preserving permutations. *Stat*.
- Ripley, B. D. (1977). Modelling spatial patterns. *JRSS-B*.
- OpenDengue — https://opendengue.org (nguồn của `dengue_dataset/`).

## Giấy phép

MIT (dự án hackathon).
