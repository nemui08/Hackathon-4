import pandas as pd

# --- Load data ---
df = pd.read_csv("cleaned_odor_data.csv")

# --- Parse time ---
df["Time"] = pd.to_datetime(df["Time"])

# --- Sensor columns ---
sensor_cols = [f"Sensor {i}" for i in range(1, 9)]

# --- บังคับเป็นตัวเลข ---
df[sensor_cols] = df[sensor_cols].apply(pd.to_numeric, errors="coerce")

# --- สร้างคอลัมน์ WEEK ---
df["Week"] = df["Time"].dt.to_period("W").astype(str)

# =========================
# 📊 ค่าเฉลี่ยรายสัปดาห์
# =========================
weekly_mean = df.groupby("Week")[sensor_cols].mean()

# --- ลดทศนิยม ---
weekly_mean = weekly_mean.round(2)

# =========================
# 💾 SAVE ทับไฟล์เดิม
# =========================
weekly_mean.to_csv("sensor_monthly_mean.csv", index=True)

print("Saved (weekly but overwritten file): sensor_monthly_mean.csv 🎉")
print(weekly_mean.head())