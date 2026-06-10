import pandas as pd

# --- 1. Load Data ---
df = pd.read_csv('Export.csv', skiprows=1)

# แปลงเวลา
df['Time'] = pd.to_datetime(df['Time'], errors='coerce')

# --- 2. เลือกคอลัมน์ sensor ---
sensor_cols = [f'Sensor {i}' for i in range(1, 9)]

# บังคับให้ sensor เป็นตัวเลข (กัน string / N/A / ช่องว่าง)
df[sensor_cols] = df[sensor_cols].apply(pd.to_numeric, errors='coerce')

# --- 3. ลบแถวที่ sensor ว่างหมด ---
df = df.dropna(subset=sensor_cols, how='all')

# --- 4. Interpolate เฉพาะตัวเลข ---
num_cols = df.select_dtypes(include='number').columns
df[num_cols] = df[num_cols].interpolate(method='linear')

# --- 5. เติมค่าข้อความ (เช่น Smell Prediction) ---
if 'Smell Prediction' in df.columns:
    df['Smell Prediction'] = df['Smell Prediction'].ffill()

# --- 6. ลบ duplicate ---
df = df.drop_duplicates(subset=['Time'], keep='first')

# --- 7. เรียงเวลา ---
df = df.sort_values(by='Time').reset_index(drop=True)

# --- 8. ลบคอลัมน์ไม่จำเป็น ---
if 'D/T' in df.columns:
    df = df.drop(columns=['D/T'])

# --- 9. เซฟไฟล์ ---
df.to_csv('Export_Cleaned.csv', index=False)

print("Data cleaning done! 🎉 Saved to Export_Cleaned.csv")