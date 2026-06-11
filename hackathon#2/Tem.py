import pandas as pd

# --- Load data ---
df = pd.read_csv("cleaned_odor_data.csv")

# --- datetime ---
df["Time"] = pd.to_datetime(df["Time"])

# --- columns ---
weather_cols = [
    "Temperature",
    "Relative Humidity",
    "Wind Speed",
    "Atmospheric Pressure"
]

# --- numeric convert ---
df[weather_cols] = df[weather_cols].apply(pd.to_numeric, errors="coerce")

# --- clean ---
df = df.dropna(subset=weather_cols)

# =========================
# 📅 WEEKLY GROUP
# =========================
df["Week"] = df["Time"].dt.to_period("W").astype(str)

weekly_weather = df.groupby("Week")[weather_cols].mean()

# =========================
# ✨ round (ลดทศนิยม)
# =========================
weekly_weather = weekly_weather.round(2)

# =========================
# 💾 save
# =========================
weekly_weather.to_csv("weekly_weather_summary.csv")

print("Saved: weekly_weather_summary.csv 🎉")
print(weekly_weather.head())