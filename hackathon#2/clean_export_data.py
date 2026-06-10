import pandas as pd

# --- 1. Load Data ---
# Skip the first row ('sep=,')
df = pd.read_csv('Export.csv', skiprows=1)

# Convert 'Time' column to datetime format
df['Time'] = pd.to_datetime(df['Time'])

# --- 2. Handle Missing Values ---
# Drop rows if all sensor columns (Sensor 1-8) are empty
sensor_cols = [f'Sensor {i}' for i in range(1, 9)]
df_cleaned = df.dropna(subset=sensor_cols, how='all')

# Use interpolate for missing numbers (sensors and weather)
df_cleaned = df_cleaned.interpolate(method='linear', numeric_only=True)

# Use forward fill (ffill) for text column (Smell Prediction)
if 'Smell Prediction' in df_cleaned.columns:
    df_cleaned['Smell Prediction'] = df_cleaned['Smell Prediction'].ffill()

# --- 3. Remove Duplicates ---
# Drop duplicate rows based on 'Time' and keep the first one
df_cleaned = df_cleaned.drop_duplicates(subset=['Time'], keep='first')

# --- 4. Sort and Save ---
# Sort data by 'Time'
df_cleaned = df_cleaned.sort_values(by='Time').reset_index(drop=True)

# Drop 'D/T' column if we don't need it
if 'D/T' in df_cleaned.columns:
    df_cleaned = df_cleaned.drop(columns=['D/T'])

# Save to a new CSV file
df_cleaned.to_csv('Export_Cleaned.csv', index=False)
print("Data cleaning is done! 🎉 Saved to 'Export_Cleaned.csv'")