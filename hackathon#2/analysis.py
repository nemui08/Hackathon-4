import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load cleaned data ---
df = pd.read_csv("Export_Cleaned.csv")

# แปลงเวลา
df["Time"] = pd.to_datetime(df["Time"])

# sensor columns
sensor_cols = [f"Sensor {i}" for i in range(1, 9)]

# =========================
# 📊 1. LINE PLOT (trend ตามเวลา)
# =========================
plt.figure()
for col in sensor_cols:
    plt.plot(df["Time"], df[col], label=col)

plt.title("Sensor Trend Over Time")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# =========================
# 📊 2. BAR (ค่าเฉลี่ยแต่ละ sensor)
# =========================
plt.figure()
df[sensor_cols].mean().plot(kind="bar")
plt.title("Average Sensor Values")
plt.ylabel("Value")
plt.tight_layout()
plt.show()


# =========================
# 📊 3. HISTOGRAM (distribution)
# =========================
plt.figure()
df[sensor_cols].hist(bins=30, figsize=(12,6))
plt.suptitle("Sensor Distribution")
plt.tight_layout()
plt.show()


# =========================
# 📊 4. BOX PLOT (ดู outlier)
# =========================
plt.figure()
sns.boxplot(data=df[sensor_cols])
plt.title("Sensor Outliers")
plt.xticks(rotation=45)
plt.show()


# =========================
# 📊 5. SCATTER (ความสัมพันธ์)
# เช่น Sensor1 vs Sensor2
# =========================
plt.figure()
plt.scatter(df["Sensor 1"], df["Sensor 2"], alpha=0.5)
plt.title("Sensor1 vs Sensor2 Relationship")
plt.xlabel("Sensor 1")
plt.ylabel("Sensor 2")
plt.show()


# =========================
# 📊 6. GROUPED (Smell Prediction vs sensor avg)
# =========================
if "Smell Prediction" in df.columns:
    grouped = df.groupby("Smell Prediction")[sensor_cols].mean()

    grouped.plot(kind="bar", figsize=(10,5))
    plt.title("Average Sensors by Smell Category")
    plt.ylabel("Value")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# =========================
# 📊 7. STACKED BAR (composition)
# =========================
top = df.head(10)[sensor_cols]

top.plot(kind="bar", stacked=True, figsize=(12,6))
plt.title("Stacked Sensor Values (First 10 rows)")
plt.tight_layout()
plt.show()