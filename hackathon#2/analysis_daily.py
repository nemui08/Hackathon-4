import pandas as pd

# --- Load data ---
df = pd.read_csv("Export_Cleaned.csv")

# --- Parse time ---
df["Time"] = pd.to_datetime(df["Time"])

# --- Convert numeric ---
df["Wind Speed"] = pd.to_numeric(df["Wind Speed"], errors="coerce")
df["PM 2.5"] = pd.to_numeric(df["PM 2.5"], errors="coerce")

# --- Clean rows ---
df = df.dropna(subset=["Time", "Wind Speed", "PM 2.5"])

# --- Create date ---
df["Date"] = df["Time"].dt.date

# =========================
# 📊 DAILY AVERAGE (2 variables only)
# =========================
daily = df.groupby("Date")[["Wind Speed", "PM 2.5"]].mean()

print(daily)

# --- save ---
daily.to_csv("daily_wind_pm.csv")

print("\nSaved: daily_wind_pm.csv 🎉")