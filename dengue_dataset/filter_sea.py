import pandas as pd

input_path = "data/releases/V1.1/Spatial_extract_V1_1.csv"
output_path = "sea_dengue_spatial.csv"

sea_countries = [
    "VIET NAM",
    "THAILAND",
    "INDONESIA",
    "MALAYSIA",
    "SINGAPORE",
    "PHILIPPINES",
    "CAMBODIA",
    "LAO PEOPLE'S DEMOCRATIC REPUBLIC",
    "MYANMAR",
    "BRUNEI DARUSSALAM",
    "TIMOR-LESTE",
]

df = pd.read_csv(input_path)

sea = df[df["adm_0_name"].isin(sea_countries)].copy()

sea["calendar_start_date"] = pd.to_datetime(sea["calendar_start_date"], errors="coerce")
sea["calendar_end_date"] = pd.to_datetime(sea["calendar_end_date"], errors="coerce")
sea["dengue_total"] = pd.to_numeric(sea["dengue_total"], errors="coerce").fillna(0)

print("SEA shape:", sea.shape)
print("\nCountries:")
print(sea["adm_0_name"].value_counts())

print("\nSpatial resolution:")
print(sea["S_res"].value_counts())

print("\nTemporal resolution:")
print(sea["T_res"].value_counts())

print("\nDate range:")
print(sea["calendar_start_date"].min(), "→", sea["calendar_end_date"].max())

print("\nSample:")
print(sea.head(10))

sea.to_csv(output_path, index=False)
print(f"\nSaved to {output_path}")