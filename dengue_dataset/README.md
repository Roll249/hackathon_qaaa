# Dengue Đông Nam Á – Tập dữ liệu đã chuẩn hoá

Bộ dữ liệu số ca sốt xuất huyết (dengue) đã được lọc và chuẩn hoá cho 11 quốc gia
Đông Nam Á, dẫn xuất từ [OpenDengue](https://opendengue.org/) (release `V1.1`).
Toàn bộ quy trình xử lý nằm trong các script Python ở thư mục gốc và đầu ra là
ba file CSV ở các mức độ chi tiết khác nhau (dài, dài đã lọc, ngang/pivot).

---

## 1. Nguồn dữ liệu

| Tham số | Giá trị |
| --- | --- |
| Dự án gốc | OpenDengue (Imperial College London) |
| Bản phát hành | `V1.1` – `data/releases/V1.1/Spatial_extract_V1_1.csv` |
| Đơn vị bệnh | Tổng số ca dengue đã báo cáo (`dengue_total`), bao gồm DF/DHF/DSS tuỳ nguồn |
| Phạm vi địa lý | 11 quốc gia Đông Nam Á (xem danh sách bên dưới) |
| Khoảng thời gian (raw) | 1955-01-01 → 2022-12-31 |
| Khoảng thời gian (chuỗi tháng Admin1) | 1993-01-01 → 2022-12-01 |
| Số nguồn (`sourceID`) chính | TYCHO 1924/2017, MOH Thái Lan, WHO/WPRO, MOH Indonesia, MOH Singapore, MOH Philippines, MOH Malaysia, … |

Cột `sourceID` cho biết bản ghi đến từ tổ chức nào (Bộ Y tế các nước, WHO khu vực
WPRO, dự án Tycho, các báo cáo y văn…). Khi cùng một quan sát có nhiều nguồn, OpenDengue
đã chọn nguồn ưu tiên trước khi đưa vào bản phát hành.

---

## 2. Các quốc gia được giữ lại

Script `filter_sea.py` lọc theo trường `adm_0_name` (viết hoa) gồm:

```
BRUNEI DARUSSALAM, CAMBODIA, INDONESIA, LAO PEOPLE'S DEMOCRATIC REPUBLIC,
MALAYSIA, MYANMAR, PHILIPPINES, SINGAPORE, THAILAND, TIMOR-LESTE, VIET NAM
```

Sau bước lọc khu vực, tập `sea_dengue_spatial.csv` chứa **69.595 dòng** ×
**14 cột**, phân bố theo quốc gia:

| Quốc gia | Số dòng |
| --- | ---: |
| THAILAND | 33.763 |
| VIET NAM | 12.564 |
| PHILIPPINES | 9.875 |
| CAMBODIA | 4.295 |
| LAO PEOPLE'S DEMOCRATIC REPUBLIC | 3.362 |
| MALAYSIA | 2.788 |
| SINGAPORE | 1.307 |
| INDONESIA | 1.277 |
| MYANMAR | 128 |
| BRUNEI DARUSSALAM | 118 |
| TIMOR-LESTE | 118 |

---

## 3. Mô tả các cột (schema)

Áp dụng cho `sea_dengue_spatial.csv` và `sea_dengue_admin1_month.csv`.

| Cột | Kiểu | Mô tả |
| --- | --- | --- |
| `adm_0_name` | str | Tên quốc gia (Admin 0, viết hoa, ví dụ `VIET NAM`). |
| `adm_1_name` | str | Tên đơn vị hành chính cấp 1 (tỉnh/bang). Rỗng nếu bản ghi ở mức quốc gia. |
| `adm_2_name` | str | Tên đơn vị hành chính cấp 2 (huyện/quận). Rỗng nếu không có. |
| `full_name` | str | Khoá địa lý kết hợp, ví dụ `VIET NAM, HA NOI`. Dùng làm cột định danh khi pivot. |
| `ISO_A0` | str | Mã ISO 3 ký tự của quốc gia (ví dụ `VNM`, `THA`). |
| `FAO_GAUL_code` | int | Mã đơn vị hành chính theo FAO GAUL. |
| `RNE_iso_code` | str | Mã ISO 3166-2 của Admin1 (ví dụ `VN-HN`). |
| `calendar_start_date` | date | Ngày bắt đầu của kỳ báo cáo (đầu tuần/tháng/năm). |
| `calendar_end_date` | date | Ngày kết thúc của kỳ báo cáo. |
| `Year` | int | Năm dương lịch của kỳ báo cáo. |
| `dengue_total` | float | Tổng số ca dengue trong kỳ báo cáo. `NaN` được điền `0`. |
| `S_res` | str | Độ phân giải không gian: `Admin0` / `Admin1` / `Admin2`. |
| `T_res` | str | Độ phân giải thời gian: `Week` / `Month` / `Year`. |
| `sourceID` | str | Định danh nguồn dữ liệu (Bộ Y tế, WHO, Tycho…). |

Phân bố độ phân giải của `sea_dengue_spatial.csv`:

- `S_res`: Admin1 = 56.362 · Admin2 = 9.060 · Admin0 = 4.173
- `T_res`: Month = 64.218 · Week = 3.623 · Year = 1.754

---

## 4. Các file dữ liệu trong repo

### 4.1. `sea_dengue_spatial.csv` – tập đầy đủ Đông Nam Á
- **Kích thước**: 69.595 dòng × 14 cột (~9,7 MB).
- **Nội dung**: tất cả bản ghi của 11 quốc gia SEA, hỗn hợp Admin0/1/2 và Week/Month/Year.
- **Sinh ra bởi**: `filter_sea.py` (lọc `Spatial_extract_V1_1.csv` theo danh sách quốc gia).
- **Mục đích**: dùng làm kho dữ liệu thô để tạo các tập con phục vụ phân tích/mô hình.

### 4.2. `sea_dengue_admin1_month.csv` – chuỗi thời gian tháng × tỉnh
- **Kích thước**: 55.030 dòng × 14 cột (~7,1 MB).
- **Lọc**: chỉ giữ `S_res == "Admin1"` và `T_res == "Month"`, loại bỏ các dòng
  thiếu `adm_0_name`, `adm_1_name`, `calendar_start_date`.
- **Sắp xếp**: theo `adm_0_name`, `adm_1_name`, `calendar_start_date`.
- **Khoảng thời gian**: 1993-01-01 → 2022-12-01.
- **Phạm vi**: **233 tỉnh/bang** tại 8 quốc gia có đủ độ phân giải Admin1/Month
  (THAILAND 33.720, VIET NAM 12.273, CAMBODIA 3.680, LAO 2.731, MALAYSIA 1.944,
  INDONESIA 421, SINGAPORE 216, TIMOR-LESTE 45).
- **Nguồn (`sourceID`)**: chủ yếu `TYCHO-ALL-1924/2017-SV_DF01` (36.646 dòng) và
  `MOH-THA-2003/2022-Y01` (18.384 dòng).
- **Sinh ra bởi**: `make_training_set.py`.

### 4.3. `sea_dengue_admin1_month_pivot.csv` – ma trận tháng × tỉnh
- **Kích thước**: 360 dòng (tháng) × 234 cột (1 cột ngày + **233 cột tỉnh**).
- **Định dạng**: dạng ngang (wide). Mỗi dòng là một tháng, mỗi cột là số ca của một tỉnh.
- **Khoá hàng**: `calendar_start_date` (đầu tháng).
- **Tên cột**: lấy từ `full_name` (ví dụ `VIET NAM, HA NOI`, `THAILAND, BANGKOK`).
- **Khoảng trống**: pivot dùng `aggfunc="sum"` và `fill_value=0`; các tỉnh chưa có
  dữ liệu trong tháng đó sẽ là `0` (không phân biệt được “0 ca” với “thiếu báo cáo”).
- **Sinh ra bởi**: `make_pivot.py`.
- **Mục đích**: phù hợp với mô hình hồi quy chuỗi thời gian đa biến, ma trận đầu vào
  cho các kỹ thuật ML/DL, hoặc trực quan hoá heatmap.

---

## 5. Quy trình xử lý (pipeline)

```
data/releases/V1.1/Spatial_extract_V1_1.csv
        │
        │ filter_sea.py            (lọc 11 quốc gia SEA, ép kiểu, fillna(0))
        ▼
sea_dengue_spatial.csv
        │
        │ make_training_set.py      (giữ Admin1 + Month, dropna, sort)
        ▼
sea_dengue_admin1_month.csv
        │
        │ make_pivot.py             (pivot rộng: thời gian × tỉnh)
        ▼
sea_dengue_admin1_month_pivot.csv
```

### Script hỗ trợ
- `inspect_data.py` – in nhanh `shape`, danh sách cột và 5 dòng đầu của ba bản
  trích xuất gốc OpenDengue (Spatial / Temporal / National).
- `check_training_set.py` – kiểm tra tập huấn luyện: số dòng, số NA theo cột,
  thống kê mô tả `dengue_total`, số tỉnh duy nhất và top 20 tỉnh có tổng số ca cao nhất.

### Cách chạy lại

```bash
# yêu cầu: Python 3.9+, pandas
python filter_sea.py           # → sea_dengue_spatial.csv
python make_training_set.py    # → sea_dengue_admin1_month.csv
python make_pivot.py           # → sea_dengue_admin1_month_pivot.csv

# (tuỳ chọn) kiểm tra chất lượng
python inspect_data.py
python check_training_set.py
```

---

## 6. Lưu ý khi sử dụng

- **Giá trị `0` mang nhiều nghĩa**: trong bản dài, `0` là số ca thực sự bằng 0;
  trong bản pivot, `0` cũng có thể có nghĩa là “tỉnh đó không có báo cáo trong tháng đó”.
  Nếu cần phân biệt, hãy quay lại `sea_dengue_admin1_month.csv` và kiểm tra sự
  hiện diện của `(adm_0_name, adm_1_name, calendar_start_date)`.
- **Không đồng nhất về nguồn**: các nước/khoảng thời gian khác nhau dùng `sourceID`
  khác nhau, kéo theo định nghĩa ca bệnh (nghi ngờ / xác nhận, DF / DHF / DSS),
  chu kỳ báo cáo và độ trễ khác nhau. Tham khảo metadata trong
  `data/metadata/` trước khi so sánh giữa các quốc gia.
- **Trùng tên Admin1**: dùng `full_name` (hoặc cặp `adm_0_name` + `adm_1_name`) làm
  khoá thay vì chỉ `adm_1_name` vì có nhiều tỉnh trùng tên giữa các quốc gia.
- **Đơn vị thời gian**: các script đã ép `calendar_start_date` / `calendar_end_date`
  thành `datetime`; tháng được biểu diễn bằng ngày đầu tháng (`YYYY-MM-01`).

---

## 7. Trích dẫn

Khi sử dụng tập dữ liệu này, vui lòng trích dẫn OpenDengue như là nguồn gốc:

> Clarke J., Lim A., Gupte P. et al. *OpenDengue: a global database of publicly
> available dengue case data.* https://opendengue.org/

Và ghi rõ phiên bản đã dùng (`V1.1`) cùng các bước lọc/biến đổi mô tả ở mục 5.
