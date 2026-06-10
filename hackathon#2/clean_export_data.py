import pandas as pd

# --- 1. Load Data ---
# Read raw CSV file and skip the metadata header row (sep=,)
df = pd.read_csv('Export.csv', skiprows=1)

# Convert Time column to standardized datetime format
df['Time'] = pd.to_datetime(df['Time'], errors='coerce')


# --- 2. Format Numeric Columns ---
# Define all columns that must contain strict numeric values
sensor_cols = [f'Sensor {i}' for i in range(1, 9)]
numeric_cols = ['D/T', 'Wind Direction', 'Wind Speed', 'Temperature', 'Relative Humidity', 'PM 2.5', 'Atmospheric Pressure'] + sensor_cols

# Enforce float/numeric data types to eliminate string or whitespace formatting errors
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')


# --- 3. Remove Completely Empty Rows ---
# Drop rows where all 8 sensor channels are missing simultaneously
df = df.dropna(subset=sensor_cols, how='all')


# --- 4. Remove System Outliers & Sensor Errors ---
# Drop corrupted rows where system errors caused impossible 0.0 values in sensors or climate metrics
# Note: Valid 0.0 values in 'Wind Speed' (calm wind) and 'D/T' (no smell) are safely preserved
df = df[
    (df['Atmospheric Pressure'] > 500) & 
    (df['Temperature'] > 0) & 
    (df['Relative Humidity'] > 0) &
    (df[sensor_cols] > 0).all(axis=1)
].copy()


# --- 5. Data Imputation ---
# Linearly interpolate missing numeric values to maintain time-series continuity
df[numeric_cols] = df[numeric_cols].interpolate(method='linear')

# Forward-fill event labels to prevent unexpected gaps in continuous smell tracking
if 'Smell Prediction' in df.columns:
    df['Smell Prediction'] = df['Smell Prediction'].ffill()
    df['Smell Prediction'] = df['Smell Prediction'].replace({'hackathon#2': 'Odor_Incident'})


# --- 6. Deduplication & Sorting ---
# Drop duplicate timestamps to ensure a strict 1-minute tracking interval
df = df.drop_duplicates(subset=['Time'], keep='first')

# Sort dataset chronologically to align the correct time-series timeline
df = df.sort_values(by='Time').reset_index(drop=True)


# --- 7. Feature Engineering ---
# Extract hour and date sub-features to optimize pattern analytics for the team
df['Hour'] = df['Time'].dt.hour
df['Date_Only'] = df['Time'].dt.date


# --- 8. Export Cleaned Pipeline Data ---
# Save the immaculate baseline dataset to feed the team's analytics pipeline
df.to_csv('cleaned_odor_data.csv', index=False)
print(f"Data preparation complete. Cleaned file saved with {len(df)} rows.")