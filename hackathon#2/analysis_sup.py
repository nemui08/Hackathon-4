import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Load Immaculate Pipeline Data ---
# Load the cleaned dataset from the subfolder path
df = pd.read_csv('hackathon#2/cleaned_odor_data.csv')

# Ensure standard datetime format for chronological continuity
df['Time'] = pd.to_datetime(df['Time'])


# --- 2. Rename Target Label (Mapping to Global Standard) ---
# Map 'hackathon#2' to 'Odor_Incident' for professional context
if 'Smell Prediction' in df.columns:
    df['Smell Prediction'] = df['Smell Prediction'].replace({'hackathon#2': 'Odor_Incident'})


# --- 3. Statistical Pattern Analysis (Ambient vs Incident) ---
print("--- 📊 Target Class Distribution ---")
print(df['Smell Prediction'].value_counts())
print("\n" + "="*50 + "\n")

print("--- 📈 Environmental & Sensor Metrics Mean Comparison ---")
# Select core features to observe behavior discrepancies during toxic smell releases
core_features = ['Wind Speed', 'Temperature', 'Relative Humidity', 'PM 2.5', 'Sensor 3', 'Sensor 5']
comparison = df.groupby('Smell Prediction')[core_features].mean().T
print(comparison)
print("\n" + "="*50 + "\n")


# --- 4. Chronological Anomaly Detection (Peak Hours) ---
print("--- ⏰ Peak Incident Occurrence Breakdown by Hour ---")
# Count the exact minutes an odor incident occurred per specific hour block
incident_hours = df[df['Smell Prediction'] == 'Odor_Incident']['Hour'].value_counts().sort_index()
print(incident_hours)


# --- 5. Data Visualization for Presentation ---
sns.set_theme(style="whitegrid")

# Graph A: Peak Incident Hours Timeline
plt.figure(figsize=(10, 5))
sns.barplot(x=incident_hours.index, y=incident_hours.values, color="crimson")
plt.title("Total Incident Duration (Minutes) Accumulated by Hour", fontsize=14, fontweight='bold')
plt.xlabel("Hour of Day (0-23)", fontsize=12)
plt.ylabel("Duration Count (Minutes)", fontsize=12)
plt.tight_layout()
plt.savefig('hackathon#2/peak_incident_hours_sup.png', dpi=300)
plt.show()

# Graph B: Feature Discrepancy Boxplot (Sensor 3 & Sensor 5 Key Drivers)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(data=df, x='Smell Prediction', y='Sensor 3', ax=axes[0], palette="Set2")
axes[0].set_title("Sensor 3 Response Profile", fontsize=12, fontweight='bold')

sns.boxplot(data=df, x='Smell Prediction', y='Sensor 5', ax=axes[1], palette="Set2")
axes[1].set_title("Sensor 5 Response Profile", fontsize=12, fontweight='bold')
plt.suptitle("Gas Sensor Behavior: Normal Air vs Odor Incident", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('hackathon#2/sensor_key_driv_sup.png', dpi=300)
plt.show()

# Graph C: Correlation Matrix During Odor Incidents
plt.figure(figsize=(8, 6))
incident_df = df[df['Smell Prediction'] == 'Odor_Incident'][core_features]
sns.heatmap(incident_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Weather & Key Sensor Correlation Matrix (During Incidents)", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('hackathon#2/incident_sup.png', dpi=300)
plt.show()

print("\n🎉 Analytics code execution finished successfully. Output graphs saved in hackathon#2 folder.")