import pandas as pd

input_path = "sea_dengue_spatial.csv"
output_path = "sea_dengue_admin1_month.csv"

df = pd.read_csv(input_path)

df["calendar_start_date"] = pd.to_datetime(df["calendar_start_date"], errors="coerce")
df["calendar_end_date"] = pd.to_datetime(df["calendar_end_date"], errors="coerce")
df["dengue_total"] = pd.to_numeric(df["dengue_total"], errors="coerce").fillna(0)

clean = df[
    (df["S_res"] == "Admin1") &
    (df["T_res"] == "Month")
].copy()

clean = clean.dropna(subset=["adm_0_name", "adm_1_name", "calendar_start_date"])

clean = clean.sort_values(
    ["adm_0_name", "adm_1_name", "calendar_start_date"]
)

print("Clean shape:", clean.shape)

print("\nCountries:")
print(clean["adm_0_name"].value_counts())

print("\nDate range:")
print(clean["calendar_start_date"].min(), "→", clean["calendar_end_date"].max())

print("\nTop regions:")
print(clean["full_name"].value_counts().head(20))

clean.to_csv(output_path, index=False)

print(f"\nSaved to {output_path}")