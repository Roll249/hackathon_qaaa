# Paradigm Shift trong Graph Construction: Từ Travel Time → Vector Transmission

## Tại sao travel time là SAI LOGIC?

**Quan sát ban đầu (sai):**
```python
# Code cũ:
speeds = np.full(n, 40.0)  # km/h
travel_time = euclid_km / speeds  # travel time của XE MÁY
adjacency = (travel_time < 60).astype(float)
```

**Vấn đề:** Code này mô tả NGƯỜI đi đường — nhưng sốt xuất huyết KHÔNG lây qua QL6!

## Cơ chế lây thật của sốt xuất huyết

### Vector chính: Aedes aegypti

| Đặc điểm | Giá trị |
|----------|---------|
| Khoảng cách bay trung bình | **105–288 m** (meta-analysis, PubMed 35640992) |
| Khoảng cách bay tối đa | **690 m** (Rio de Janeiro study) |
| 90% cá thể bay | **< 500 m** |
| Bán kính vector control (WHO) | **200–400 m** |
| Thế hệ (generation time) | ~2–3 tuần |

→ **Scale quan trọng: METERS, không phải KILOMETERS**

### Vector phụ: Con người (human mobility)

- Bệnh nhân mang virus đi từ xã A → xã B qua xe khách, máy bay
- Đây là **exogenous forcing** (lực ngoại sinh), không phải nội sinh
- Xảy ra ở scale vài chục km hoặc hơn, NHƯNG yếu (weight thấp)

## Fix mới

```python
# Code mới:
VECTOR_MEAN_DISPERSAL_M = 200.0
transmission = np.exp(-euclid_m / VECTOR_MEAN_DISPERSAL_M)
adjacency = (euclid_m < 5000).astype(float) * transmission
# + sparse exogenous edges cho human mobility
```

### Các thay đổi cụ thể

| Component | Cũ (sai) | Mới (đúng) |
|-----------|----------|------------|
| Edge weight | travel time (giờ) | transmission strength [0,1] |
| Distance scale | kilometers | **meters** |
| Threshold | 60 giờ (~không lọt) | **5 km** (neighborhood scale) |
| Kernel | 1/speed | **exp(-d/200m)** |
| Risk intensity | random + hotspots | **epidemiology formula** |
| Graph structure | fully connected (mean degree 129) | **sparse** (mean degree ~3) |

## Risk intensity mới — công thức epidemiology

```python
risk[i] = (
    0.30 * log_cases       # Lịch sử ca bệnh (logarithmic)
  + 0.30 * altitude × water  # Vector capacity
  + 0.20 * pop_density    # Mật độ dân số
  + 0.20 * housing        # Chất lượng nhà (containers)
)
```

| Factor | Logic |
|--------|------|
| **Historical cases** | Log scale: ca cao → nguy cơ cao (nhưng diminishing returns) |
| **Altitude** | Aedes thích < 1500m (Điện Biên 200-1800m) |
| **Stagnant water** | Vector breeding sites — yếu tố quyết định |
| **Population density** | Mật độ cao → nhiều human host |
| **Housing quality** | Nhà xấu → nhiều containers → nhiều breeding |

## Kết quả sau fix

```
Mean degree: 129 (fully connected) → 2.86 (sparse, realistic)
Edges: 8385 → 186 (neighborhood-level)
Risk intensity: random → epidemiology formula
Risk range: [0.229, 0.958] — meaningful spread
```

## Tại sao fix này KHÔNG phải detail nhỏ?

Đây là **đổi tư duy thứ 2** trong project:

1. **Paradigm shift 1:** Predict propagation → Find peaks (cá leo cây)
2. **Paradigm shift 2:** Travel time → Vector transmission (biology)

Nếu graph là fully connected, LQW và quantum walk **trở nên vô nghĩa** — không có neighborhood structure để exploit.

Với sparse graph + vector kernel, LQW thực sự **propagate theo mạng lưới lây nhiễm** — và quantum advantage O(√N) trở nên có ý nghĩa.

## Reference

- Aedes aegypti flight distance: PubMed 35640992 (meta-analysis 27 experiments)
- Rio dispersal study: Honório et al. 2009, doi:10.1590/s0034-89102009000100002
- ECDC factsheet: https://www.ecdc.europa.eu/en/disease-vectors/facts/mosquito-factsheets/aedes-aegypti
- WHO vector control guidelines: 200-400m radius around case index

## Kết luận

> **"Dịch nó lan truyền khác với người đi"** — bro đã đúng.

Graph giờ phản ánh **cơ chế sinh học của vector**, không phải **hành vi của con người**. Đây là một trong những điểm quan trọng nhất của quantum epidemiology: phải hiểu **cấu trúc thật** của epidemic spread, không ép model vào data structure tiện tay.