import pandas as pd

path = "sea_dengue_admin1_month.csv"
df = pd.read_csv(path, low_memory=False)

df["calendar_start_date"] = pd.to_datetime(df["calendar_start_date"], errors="coerce")
df["dengue_total"] = pd.to_numeric(df["dengue_total"], errors="coerce").fillna(0)

pivot = df.pivot_table(
    index="calendar_start_date",
    columns="full_name",
    values="dengue_total",
    aggfunc="sum",
    fill_value=0
)

pivot = pivot.sort_index()

print("Pivot shape:", pivot.shape)
print(pivot.head())

pivot.to_csv("sea_dengue_admin1_month_pivot.csv")
print("\nSaved to sea_dengue_admin1_month_pivot.csv")