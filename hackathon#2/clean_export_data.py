import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ==================================================
# 0) PATH SETUP (FIX FILE NOT FOUND ISSUE)
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Export.csv"

print("Loading data...")

df = pd.read_csv(
    DATA_PATH,
    skiprows=1
)

print(f"Rows before cleaning: {len(df):,}")

# ==================================================
# 1) TIME CONVERSION
# ==================================================

if "Time" in df.columns:
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")

# ==================================================
# 2) SENSOR COLUMNS
# ==================================================

sensor_cols = [f"Sensor {i}" for i in range(1, 9)]
sensor_cols = [c for c in sensor_cols if c in df.columns]

# ==================================================
# 3) REMOVE EMPTY SENSOR ROWS
# ==================================================

if sensor_cols:
    df = df.dropna(subset=sensor_cols, how="all")

print(f"Rows after removing empty rows: {len(df):,}")

# ==================================================
# 4) CONVERT NUMERIC
# ==================================================

numeric_candidates = [
    "D/T",
    "Wind Direction",
    "Wind Speed",
    "Temperature",
    "Relative Humidity",
    "PM 2.5",
    "Atmospheric Pressure"
] + sensor_cols

for col in numeric_candidates:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ==================================================
# 5) INTERPOLATION
# ==================================================

numeric_cols = df.select_dtypes(include=["number"]).columns

df[numeric_cols] = df[numeric_cols].interpolate(method="linear")

# ==================================================
# 6) TEXT COLUMN SAFE HANDLING
# ==================================================

if "Smell Prediction" in df.columns:
    df["Smell Prediction"] = df["Smell Prediction"].ffill()

# ==================================================
# 7) REMOVE DUPLICATES + SORT
# ==================================================

if "Time" in df.columns:
    df = df.drop_duplicates(subset=["Time"])
    df = df.sort_values("Time").reset_index(drop=True)

# ==================================================
# 8) ODOR INDEX
# ==================================================

if sensor_cols:
    df["Odor_Index"] = df[sensor_cols].mean(axis=1)

# ==================================================
# 9) SAVE CLEAN DATA
# ==================================================

CLEAN_PATH = BASE_DIR / "Export_Cleaned.csv"
df.to_csv(CLEAN_PATH, index=False)

print(f"Saved: {CLEAN_PATH}")

# ==================================================
# 10) ODOR TREND
# ==================================================

if "Time" in df.columns and "Odor_Index" in df.columns:
    plt.figure(figsize=(15, 5))
    plt.plot(df["Time"], df["Odor_Index"])
    plt.title("Odor Index Over Time")
    plt.xlabel("Time")
    plt.ylabel("Odor Index")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ==================================================
# 11) HOURLY ANALYSIS
# ==================================================

if "Time" in df.columns and "Odor_Index" in df.columns:
    df["Hour"] = df["Time"].dt.hour

    hourly = df.groupby("Hour")["Odor_Index"].mean()

    plt.figure(figsize=(10, 5))
    hourly.plot(kind="bar")
    plt.title("Average Odor by Hour")
    plt.ylabel("Odor Index")
    plt.tight_layout()
    plt.show()

# ==================================================
# 12) DAILY ANALYSIS
# ==================================================

if "Time" in df.columns and "Odor_Index" in df.columns:
    daily = df.resample("D", on="Time")["Odor_Index"].mean()

    plt.figure(figsize=(15, 5))
    daily.plot()
    plt.title("Daily Average Odor Index")
    plt.ylabel("Odor Index")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ==================================================
# 13) CORRELATION
# ==================================================

weather_cols = [
    "Wind Speed",
    "Wind Direction",
    "Temperature",
    "Relative Humidity",
    "PM 2.5",
    "Atmospheric Pressure",
    "Odor_Index"
]

available_cols = [c for c in weather_cols if c in df.columns]

if len(available_cols) > 1:
    corr = df[available_cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()

    print("\nCorrelation with Odor_Index\n")
    print(corr["Odor_Index"].sort_values(ascending=False))

# ==================================================
# 14) WIND ANALYSIS
# ==================================================

if "Wind Direction" in df.columns and "Odor_Index" in df.columns:

    bins = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

    df["Wind_Sector"] = pd.cut(
        df["Wind Direction"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    wind_avg = df.groupby("Wind_Sector")["Odor_Index"].mean()

    plt.figure(figsize=(8, 5))
    wind_avg.plot(kind="bar")
    plt.title("Average Odor by Wind Direction")
    plt.ylabel("Odor Index")
    plt.tight_layout()
    plt.show()

# ==================================================
# 15) ANOMALY DETECTION
# ==================================================

if "Odor_Index" in df.columns:

    mean = df["Odor_Index"].mean()
    std = df["Odor_Index"].std()
    threshold = mean + (3 * std)

    anomaly = df[df["Odor_Index"] > threshold]

    print("\n========== ANOMALY REPORT ==========")
    print(f"Mean      : {mean:.2f}")
    print(f"Std       : {std:.2f}")
    print(f"Threshold : {threshold:.2f}")
    print(f"Anomalies : {len(anomaly)}")

    ANOMALY_PATH = BASE_DIR / "Odor_Anomaly.csv"
    anomaly.to_csv(ANOMALY_PATH, index=False)

# ==================================================
# 16) ANOMALY PLOT
# ==================================================

if "Odor_Index" in df.columns and "Time" in df.columns:

    plt.figure(figsize=(15, 5))
    plt.plot(df["Time"], df["Odor_Index"], label="Odor Index")

    if len(anomaly) > 0:
        plt.scatter(anomaly["Time"], anomaly["Odor_Index"], s=20, label="Anomaly")

    plt.axhline(threshold, linestyle="--", label="Threshold")
    plt.legend()
    plt.title("Odor Anomaly Detection")
    plt.tight_layout()
    plt.show()

# ==================================================
# 17) TOP EVENTS
# ==================================================

if "Odor_Index" in df.columns:

    top10 = df.sort_values(by="Odor_Index", ascending=False).head(10)

    TOP_PATH = BASE_DIR / "Top10_Odor_Events.csv"
    top10.to_csv(TOP_PATH, index=False)

    print("\nTop 10 odor events")
    print(top10[["Time", "Odor_Index"]])

print("\nAnalysis Complete ✅")