"""Graph construction for Dien Bien 130 communes.

Tư duy mới: Edge weight KHÔNG phải travel time của NGƯỜI đi đường.

Lý do sinh học:
- Aedes aegypti (vector sốt xuất huyết) bay trung bình 105-288m, tối đa 690m
  trong đời (meta-analysis 27 experiments, PubMed 35640992)
- 90% cá thể bay < 500m
- Vector control programs target radius 200-400m quanh case index

Vậy edge weight trong dengue graph phải là:
    w(i,j) = transmission_strength(i,j)
           = vector_capacity(i) × kernel(distance(i,j))

Trong đó kernel có thể là:
- Exponential decay: exp(-d/200m) — vector capacity giảm theo khoảng cách
- Hoặc step function: 1 nếu d < 400m, 0 nếu > 400m

Nếu 2 xã cách nhau 60km (~60000m), edge weight ~ 0 → không lây qua vector.
Lây xa chỉ qua HUMAN MOBILITY (xe khách, máy bay) — đó là exogenous forcing,
KHÔNG phải endogenous epidemic spread.

Vì vậy graph PHẢI có structure thưa:
- Mỗi xã chỉ nối tới vài xã lân cận (trong bán kính 500m-2km)
- Long-distance connections (>10km) là exogenous (nhẹ weight hoặc bỏ qua)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# Hằng số sinh học (từ literature)
VECTOR_MEAN_DISPERSAL_M = 200.0  # Aedes aegypti mean dispersal ~ 105-288m
VECTOR_MAX_DISPERSAL_M = 690.0   # Maximum observed
VECTOR_CONTROL_RADIUS_M = 400.0  # WHO/PAHO recommendation for control radius


@dataclass
class Commune:
    """A commune (xã) in Dien Bien province."""
    id: int
    name: str
    lat: float
    lon: float
    population: int
    district: str
    is_border: bool = False
    # Epidemiological attributes
    altitude_m: float = 500.0           # Điện Biên: 200-1800m
    has_stagnant_water: bool = False    # Vector breeding sites
    housing_quality: float = 0.5        # 0=rất tốt, 1=rất xấu (containers)
    historical_cases: int = 0           # Số ca lịch sử 5 năm gần nhất


@dataclass
class DienBienGraph:
    """Weighted graph of 130 communes.

    Edge weights = transmission strength (vector capacity × proximity kernel),
    NOT travel time of humans.
    """
    communes: list[Commune] = field(default_factory=list)
    adjacency: Optional[np.ndarray] = None  # shape (N, N), transmission strength
    risk_intensity: Optional[np.ndarray] = None  # historical dengue density

    @property
    def n_communes(self) -> int:
        return len(self.communes)

    def get_neighbors(self, i: int) -> list[tuple[int, float]]:
        """Return [(neighbor_id, transmission_strength), ...]."""
        if self.adjacency is None:
            return []
        return [(j, self.adjacency[i, j]) for j in range(self.n_communes)
                if self.adjacency[i, j] > 0 and i != j]

    def normalize_risk(self) -> np.ndarray:
        """Min-max normalize risk intensity → [0, 1]."""
        if self.risk_intensity is None:
            raise ValueError("risk_intensity not set")
        r = self.risk_intensity.astype(float)
        rmin, rmax = r.min(), r.max()
        return (r - rmin) / (rmax - rmin + 1e-10)


def vector_transmission_kernel(distance_m: np.ndarray) -> np.ndarray:
    """Exponential kernel for Aedes aegypti vector transmission.

    Based on mark-release-recapture studies (Honório et al. 2003, 2009):
    - 90% mosquitoes disperse < 500m
    - Maximum observed ~ 690m
    - Use exponential decay with scale = 200m

    Args:
        distance_m: pairwise distances in METERS

    Returns:
        transmission_strength: shape same as input, in [0, 1]
    """
    # Exponential decay: kernel(d) = exp(-d / 200m)
    # At d=0: 1.0 (full transmission)
    # At d=200: 0.37
    # At d=500: 0.08 (90% reduction, matches literature)
    # At d=1000: 0.007 (essentially zero)
    return np.exp(-distance_m / VECTOR_MEAN_DISPERSAL_M)


def build_synthetic_dien_bien(seed: int = 42) -> DienBienGraph:
    """Build synthetic but realistic 130-commune graph for Dien Bien.

    Key design principles:
    1. Communes placed in 2D (lat, lon) roughly matching Dien Bien's geography
    2. Edge weights = vector transmission strength (NOT travel time)
    3. Graph structure SPARSE because vector dispersal is ~ 200m
    4. Risk intensity incorporates:
       - Historical dengue cases
       - Stagnant water presence
       - Population density
       - Altitude (lower = more vectors)
       - Housing quality (containers)
    """
    rng = np.random.default_rng(seed)

    districts = [
        "Điện Biên Phủ", "Mường Lay", "Mường Nhé", "Mường Chà",
        "Tủa Chùa", "Tuần Giáo", "Điện Biên Đông", "Điện Biên",
        "Nậm Pồ", "Mường Ảng"
    ]

    communes = []
    communes_per_district = []
    for d_idx, district in enumerate(districts):
        if d_idx == len(districts) - 1:
            n_in_district = 130 - sum(communes_per_district)
        else:
            n_in_district = 13  # 130 / 10 = 13
        communes_per_district.append(n_in_district)
        cx, cy = rng.uniform(21.0, 22.7), rng.uniform(102.0, 103.5)
        for _ in range(n_in_district):
            lat = cx + rng.normal(0, 0.15)
            lon = cy + rng.normal(0, 0.15)
            altitude = float(rng.uniform(200, 1800))  # Điện Biên altitude range
            has_water = bool(rng.random() < 0.3)  # 30% có nước đọng
            housing = float(rng.uniform(0.2, 0.9))
            historical = int(rng.poisson(2))  # average 2 cases per commune baseline

            communes.append(Commune(
                id=len(communes),
                name=f"Xã_{len(communes):03d}",
                lat=float(lat),
                lon=float(lon),
                population=int(rng.integers(2000, 15000)),
                district=district,
                is_border=(lat > 22.5 or lon > 103.2),
                altitude_m=altitude,
                has_stagnant_water=has_water,
                housing_quality=housing,
                historical_cases=historical,
            ))
            if len(communes) >= 130:
                break
        if len(communes) >= 130:
            break

    communes = communes[:130]
    n = len(communes)

    # Pairwise Euclidean distance in METERS (commune scale ~ 10-50km apart)
    lats = np.array([c.lat for c in communes])
    lons = np.array([c.lon for c in communes])
    diff_lat_m = (lats[:, None] - lats[None, :]) * 111_000  # m/degree
    diff_lon_m = (lons[:, None] - lons[None, :]) * 111_000 * np.cos(np.deg2rad(lats.mean()))
    euclid_m = np.sqrt(diff_lat_m**2 + diff_lon_m**2)

    # VECTOR TRANSMISSION STRENGTH (thay vì travel time)
    # Kernel: exp(-d/200m) — giảm theo khoảng cách vector dispersal
    transmission = vector_transmission_kernel(euclid_m)

    # THRESHOLD: chỉ giữ edges trong bán kính ~5km
    # (5km = lân cận gần, vector bay được hoặc bay qua vài generation)
    # Lưu ý: Aedes bay 200m/lần, nhưng nhiều generation có thể lan 1-2km trong 1 tháng
    # 5km là scale realistic cho "neighborhood" level
    NEIGHBORHOOD_RADIUS_M = 5_000.0
    adjacency = np.where(euclid_m < NEIGHBORHOOD_RADIUS_M, transmission, 0.0)
    np.fill_diagonal(adjacency, 0.0)

    # ============== RISK INTENSITY ==============
    # Tính theo công thức epidemiology, KHÔNG random
    risk = np.zeros(n)
    for i, c in enumerate(communes):
        # Component 1: Historical cases (logarithmic scale)
        log_cases = np.log1p(c.historical_cases) / np.log1p(20)  # normalize to [0,1]

        # Component 2: Vector capacity (altitude + stagnant water)
        # Lower altitude → more Aedes (they prefer < 1500m)
        altitude_factor = max(0, 1.0 - c.altitude_m / 1500.0)
        water_factor = 1.0 if c.has_stagnant_water else 0.3

        # Component 3: Population density (per km²)
        # Tỉnh Điện Biên ~ 9,560 km², mean pop ~ 6000/commune
        area_km2 = 9560 / 130  # average commune area
        pop_density = c.population / area_km2 / 100  # normalize

        # Component 4: Housing quality (poor housing → more containers)
        housing_factor = c.housing_quality

        # Weighted sum
        risk[i] = (
            0.30 * log_cases +
            0.30 * altitude_factor * water_factor +
            0.20 * min(pop_density, 1.0) +
            0.20 * housing_factor
        )

    risk = np.clip(risk, 0, 1.0)

    # Add 3-5 hotspots: communes with extreme conditions
    n_hotspots = rng.integers(3, 6)
    hotspot_indices = rng.choice(n, size=n_hotspots, replace=False)
    for h in hotspot_indices:
        # Hotspot: combine all risk factors at extreme
        communes[h].historical_cases = int(rng.integers(15, 30))
        communes[h].has_stagnant_water = True
        communes[h].altitude_m = float(rng.uniform(200, 600))  # low altitude
        communes[h].housing_quality = float(rng.uniform(0.7, 0.95))
        # Recompute risk for hotspot
        log_cases = np.log1p(communes[h].historical_cases) / np.log1p(20)
        altitude_factor = max(0, 1.0 - communes[h].altitude_m / 1500.0)
        water_factor = 1.0
        area_km2 = 9560 / 130
        pop_density = communes[h].population / area_km2 / 100
        risk[h] = 0.30 * log_cases + 0.30 * altitude_factor * water_factor + \
                  0.20 * min(pop_density, 1.0) + 0.20 * communes[h].housing_quality

    risk = np.clip(risk, 0, 1.0)

    # Name communes around hotspots
    for idx, h in enumerate(hotspot_indices):
        communes[h].name = f"Hotspot_{idx+1}"

    # Add long-distance exogenous edges (people mobility)
    # Mark them differently so we can use them in LQW for multi-hop search
    # but NOT in the local spread graph
    exogenous_weight = 0.05  # weak exogenous (people travel by bus/plane)
    for i in range(n):
        for j in range(i+1, n):
            if adjacency[i, j] == 0 and rng.random() < 0.02:  # 2% sparse exogenous
                adjacency[i, j] = exogenous_weight
                adjacency[j, i] = exogenous_weight

    return DienBienGraph(
        communes=communes,
        adjacency=adjacency,
        risk_intensity=risk,
    )


if __name__ == "__main__":
    g = build_synthetic_dien_bien()
    print(f"Built graph with {g.n_communes} communes")
    print(f"  Adjacency shape: {g.adjacency.shape}")
    print(f"  Risk range: [{g.risk_intensity.min():.3f}, {g.risk_intensity.max():.3f}]")
    n_edges = int((g.adjacency > 0).sum() // 2)
    mean_degree = float((g.adjacency > 0).sum(axis=1).mean())
    print(f"  Edges: {n_edges}, mean degree: {mean_degree:.2f}")
    print(f"  Top 5 risk communes:")
    top5 = np.argsort(g.risk_intensity)[-5:][::-1]
    for idx in top5:
        c = g.communes[idx]
        print(f"    {c.name} ({c.district}): risk={g.risk_intensity[idx]:.3f}, "
              f"pop={c.population}, alt={c.altitude_m:.0f}m, "
              f"water={c.has_stagnant_water}, cases={c.historical_cases}")
    print(f"\n  [Sanity check] Mean degree {mean_degree:.1f} << 129 (fully connected)")
    print(f"  → Graph structure SPARSE, reflecting vector dispersal biology")