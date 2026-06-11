import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

# ==================================================
# LOAD DATA
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "cleaned_odor_data.csv"

print("Loading data...")

df = pd.read_csv(DATA_PATH)

# ==================================================
# TIME CONVERSION
# ==================================================

df["Time"] = pd.to_datetime(
    df["Time"],
    errors="coerce"
)

df = df.dropna(subset=["Time"])

print(f"Rows loaded: {len(df):,}")

# ==================================================
# D/T CHECK
# ==================================================

if "D/T" not in df.columns:
    raise ValueError(
        "Column 'D/T' not found."
    )

print("\nD/T Statistics")
print(df["D/T"].describe())

# ==================================================
# DYNAMIC ODOR THRESHOLD
# ==================================================

ODOR_THRESHOLD = df["D/T"].quantile(0.95)

print(
    f"\nOdor Incident Threshold (95th percentile): "
    f"{ODOR_THRESHOLD:.2f}"
)

# ==================================================
# INCIDENT FLAG
# ==================================================

df["Odor_Incident"] = np.where(
    df["D/T"] >= ODOR_THRESHOLD,
    "Odor",
    "Normal"
)

incident_df = df[
    df["D/T"] >= ODOR_THRESHOLD
].copy()

print(
    f"Odor Incident Rows: "
    f"{len(incident_df):,}"
)

# ==================================================
# COMMUNITY RISK INDEX
# ==================================================

def community_risk(row):

    score = 0

    if row["D/T"] >= ODOR_THRESHOLD:
        score += 3

    if (
        "Relative Humidity" in df.columns
        and row["Relative Humidity"] > 80
    ):
        score += 1

    if (
        "Wind Speed" in df.columns
        and row["Wind Speed"] < 2
    ):
        score += 1

    if score >= 5:
        return "Red"

    elif score >= 3:
        return "Yellow"

    return "Green"

df["Community_Risk"] = df.apply(
    community_risk,
    axis=1
)

# ==================================================
# PEAK INCIDENT TIMELINES
# ==================================================

incident_df["Hour"] = (
    incident_df["Time"].dt.hour
)

peak_hours = (
    incident_df.groupby("Hour")
    .size()
    .sort_values(ascending=False)
)

print("\n========== PEAK INCIDENT HOURS ==========")

if len(peak_hours) > 0:

    print(peak_hours.head(10))

    plt.figure(figsize=(10, 5))

    peak_hours.sort_index().plot(
        kind="bar"
    )

    plt.title(
        "Peak Incident Hours"
    )

    plt.xlabel(
        "Hour of Day"
    )

    plt.ylabel(
        "Incident Count"
    )

    plt.tight_layout()

    plt.savefig(
        BASE_DIR / "peak_incident_hours.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# ==================================================
# SENSOR 3 / SENSOR 5 BOXPLOT
# ==================================================

if (
    "Sensor 3" in df.columns
    and "Sensor 5" in df.columns
):

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5)
    )

    sns.boxplot(
        data=df,
        x="Odor_Incident",
        y="Sensor 3",
        ax=axes[0]
    )

    axes[0].set_title(
        "Sensor 3 Response"
    )

    sns.boxplot(
        data=df,
        x="Odor_Incident",
        y="Sensor 5",
        ax=axes[1]
    )

    axes[1].set_title(
        "Sensor 5 Response"
    )

    plt.tight_layout()

    plt.savefig(
        BASE_DIR / "sensor_key_drivers.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# ==================================================
# CORRELATION HEATMAP
# ==================================================

corr_cols = [
    "D/T",
    "Sensor 3",
    "Sensor 5",
    "Wind Speed",
    "Temperature",
    "Relative Humidity",
    "PM 2.5"
]

corr_cols = [
    c for c in corr_cols
    if c in incident_df.columns
]

if len(corr_cols) > 1:

    corr_matrix = (
        incident_df[corr_cols]
        .corr()
    )

    plt.figure(
        figsize=(10, 8)
    )

    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title(
        "Correlation Matrix During Odor Incidents"
    )

    plt.tight_layout()

    plt.savefig(
        BASE_DIR / "incident_correlation.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# ==================================================
# WIND ANALYSIS
# ==================================================

worst_wind = "Unknown"

if "Wind Direction" in incident_df.columns:

    bins = [
        0,45,90,135,
        180,225,270,
        315,360
    ]

    labels = [
        "N","NE","E","SE",
        "S","SW","W","NW"
    ]

    incident_df["Wind_Sector"] = pd.cut(
        incident_df["Wind Direction"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    if len(incident_df) > 0:

        worst_wind = str(
            incident_df["Wind_Sector"]
            .mode()[0]
        )

# ==================================================
# RISK SUMMARY CSV
# ==================================================

risk_export = df[
    [
        "Time",
        "D/T",
        "Community_Risk"
    ]
]

risk_export.to_csv(
    BASE_DIR / "Community_Risk_Index.csv",
    index=False
)

# ==================================================
# INSIGHT SUMMARY JSON
# ==================================================

summary = {

    "Threshold_D_T": float(
        ODOR_THRESHOLD
    ),

    "Incident_Count": int(
        len(incident_df)
    ),

    "Risk_Distribution":

        df["Community_Risk"]
        .value_counts()
        .to_dict(),

    "Top_Peak_Hours":

        peak_hours.head(5)
        .index.tolist(),

    "Worst_Wind_Direction":

        worst_wind
}

if (
    "Wind Speed" in incident_df.columns
    and len(incident_df) > 0
):
    summary[
        "Average_Wind_Speed"
    ] = round(
        float(
            incident_df["Wind Speed"]
            .mean()
        ),
        2
    )

if (
    "Relative Humidity"
    in incident_df.columns
    and len(incident_df) > 0
):
    summary[
        "Average_Humidity"
    ] = round(
        float(
            incident_df[
                "Relative Humidity"
            ].mean()
        ),
        2
    )

if (
    "Temperature"
    in incident_df.columns
    and len(incident_df) > 0
):
    summary[
        "Average_Temperature"
    ] = round(
        float(
            incident_df[
                "Temperature"
            ].mean()
        ),
        2
    )

with open(
    BASE_DIR / "Insight_Summary.json",
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )

# ==================================================
# FILE SUMMARY
# ==================================================

print("\n========== FILES GENERATED ==========")

print("peak_incident_hours.png")
print("sensor_key_drivers.png")
print("incident_correlation.png")
print("Community_Risk_Index.csv")
print("Insight_Summary.json")

print("\nAnalysis Complete ✅")