import pandas as pd

spatial_path = "data/releases/V1.1/Spatial_extract_V1_1.csv"
temporal_path = "data/releases/V1.1/Temporal_extract_V1_1.csv"
national_path = "data/releases/V1.1/National_extract_V1_1.csv"

for name, path in [
    ("SPATIAL", spatial_path),
    ("TEMPORAL", temporal_path),
    ("NATIONAL", national_path),
]:
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    df = pd.read_csv(path)

    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nHead:")
    print(df.head(5))