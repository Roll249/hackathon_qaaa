import pandas as pd

path = "sea_dengue_admin1_month.csv"
df = pd.read_csv(path, low_memory=False)

print("Shape:", df.shape)

print("\nMissing values:")
print(df.isna().sum())

print("\nDengue total summary:")
print(df["dengue_total"].describe())

print("\nRows with dengue_total > 0:")
print((df["dengue_total"] > 0).sum())

print("\nNumber of unique regions:")
print(df["full_name"].nunique())

print("\nTop 20 regions by total dengue:")
print(
    df.groupby("full_name")["dengue_total"]
    .sum()
    .sort_values(ascending=False)
    .head(20)
)