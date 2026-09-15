import pandas as pd
import numpy as np

# Load processed dataset
df = pd.read_csv("../data/FD001_train_processed.csv")

print("Dataset shape:", df.shape)
print(df.head())
print(df.columns.tolist())
sensor_columns = [col for col in df.columns if col.startswith("sensor_")]

print("Number of sensors:", len(sensor_columns))
print(sensor_columns)
for sensor in sensor_columns:
    df[f"{sensor}_rolling_mean"] = (
        df.groupby("unit")[sensor]
        .transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    )

    df[f"{sensor}_rolling_std"] = (
        df.groupby("unit")[sensor]
        .transform(lambda x: x.rolling(window=5, min_periods=1).std())
    )
df = df.fillna(0)
df["cycle_ratio"] = df["cycle"] / df["max_cycle"]
print("Final dataset shape:", df.shape)
print(df.head())
df.to_csv("../data/FD001_feature_engineered.csv", index=False)

print("Feature engineered dataset saved successfully.")