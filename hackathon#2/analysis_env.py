import pandas as pd

# --- Load data ---
df = pd.read_csv("Export_Cleaned.csv")

# --- Parse time ---
df["Time"] = pd.to_datetime(df["Time"])

# --- Sensor columns ---
sensor_cols = [f"Sensor {i}" for i in range(1, 9)]

# --- บังคับเป็นตัวเลข ---
df[sensor_cols] = df[sensor_cols].apply(pd.to_numeric, errors="coerce")

# --- สร้างคอลัมน์เดือน ---
df["Month"] = df["Time"].dt.to_period("M")

# =========================
# 📊 ค่าเฉลี่ยรายเดือน
# =========================
monthly_mean = df.groupby("Month")[sensor_cols].mean()

print(monthly_mean)

# =========================
# 💾 save ไฟล์ (สำคัญมากสำหรับส่งงาน)
# =========================
monthly_mean.to_csv("sensor_monthly_mean.csv")

print("\nSaved: sensor_monthly_mean.csv 🎉")