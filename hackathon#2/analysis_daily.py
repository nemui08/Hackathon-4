import pandas as pd

# --- Load data ---
df = pd.read_csv("cleaned_odor_data.csv")

# --- datetime ---
df["Time"] = pd.to_datetime(df["Time"])

# --- numeric convert ---
cols = ["D/T", "Wind Speed", "PM 2.5"]
df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")

df = df.dropna(subset=cols)

# =========================
# 📅 WEEKLY GROUP
# =========================
df["Week"] = df["Time"].dt.to_period("W").astype(str)

weekly = df.groupby("Week")[cols].mean().reset_index()

# =========================
# ✨ CLEAN FORMAT (ลดทศนิยม)
# =========================
weekly["D/T"] = weekly["D/T"].round(2)
weekly["Wind Speed"] = weekly["Wind Speed"].round(2)
weekly["PM 2.5"] = weekly["PM 2.5"].round(2)

# =========================
# 💾 SAVE
# =========================
weekly.to_csv("weekly_summary.csv", index=False)

print("Saved: weekly_summary.csv 🎉")
print(weekly.head())