# ================================
# WEARABLE AI PROJECT - FULL PIPELINE
# ================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ================================
# 1. LOAD SIGNAL FUNCTION
# ================================x
def load_signal(file_path, col_names):
    df = pd.read_csv(file_path, header=None)

    start_time = df.iloc[0, 0]
    sampling_rate = df.iloc[1, 0]

    df = df.iloc[2:].reset_index(drop=True)

    if isinstance(col_names, list):
        df.columns = col_names
    else:
        df.columns = [col_names]

    # FIX: convert to milliseconds
    freq_ms = int((1 / sampling_rate) * 1000)

    df['time'] = pd.date_range(
        start=pd.to_datetime(start_time, unit='s'),
        periods=len(df),
        freq=f"{freq_ms}ms"
    )

    return df


# ================================
# 2. LOAD ALL FILES
# ================================
eda = load_signal("EDA.csv", "EDA")
temp = load_signal("TEMP.csv", "TEMP")
hr = load_signal("HR.csv", "HR")
bvp = load_signal("BVP.csv", "BVP")

# ACC (3-axis)
acc = load_signal("ACC.csv", ['x', 'y', 'z'])

print("All files loaded successfully!")


# ================================
# 3. MERGE DATA
# ================================
df = eda.merge(temp, on='time', how='inner')
df = df.merge(hr, on='time', how='inner')
df = df.merge(acc, on='time', how='inner')

df = df.sort_values(by='time').reset_index(drop=True)

print("Data merged successfully!")


# ================================
# 4. FEATURE ENGINEERING
# ================================

# ACC features (movement)
df['acc_magnitude'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
df['activity_mean'] = df['acc_magnitude'].rolling(50).mean()

# HR features
df['hr_mean'] = df['HR'].rolling(10).mean()
df['hr_std'] = df['HR'].rolling(10).std()

# HRV (approx using HR variation)
df['hrv'] = df['HR'].rolling(10).std()

# EDA features (stress)
df['eda_mean'] = df['EDA'].rolling(10).mean()
df['eda_std'] = df['EDA'].rolling(10).std()

# TEMP features
df['temp_mean'] = df['TEMP'].rolling(50).mean()

print("Feature engineering done!")


# ================================
# 5. CREATE LABELS
# ================================
# NOTE: Replace this with tags.csv later

df['label'] = 0

# Simple stress rule (temporary)
df.loc[
    (df['eda_mean'] > df['eda_mean'].quantile(0.75)) &
    (df['hr_mean'] > df['hr_mean'].quantile(0.60)),
    'label'
] = 1

print("Labels created!")


# ================================
# 6. CLEAN DATA
# ================================
df = df.dropna().reset_index(drop=True)

print("Data cleaned!")


# ================================
# 7. SELECT FEATURES
# ================================
features = [
    'acc_magnitude',
    'activity_mean',
    'hr_mean',
    'hr_std',
    'eda_mean',
    'eda_std',
    'temp_mean',
    'hrv'
]

X = df[features]
y = df['label']

print("Features selected!")


# ================================
# 8. TRAIN-TEST SPLIT
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Data split done!")


# ================================
# 9. TRAIN MODEL
# ================================
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

print("Model trained!")


# ================================
# 10. EVALUATE MODEL
# ================================
y_pred = model.predict(X_test)

print("\n===== MODEL PERFORMANCE =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# ================================
# 11. RISK SCORE (PROBABILITY)
# ================================
sample = X_test.iloc[0:1]
prob = model.predict_proba(sample)

print("\nSample Prediction:", model.predict(sample))
print("Risk Score (probability):", prob)


# ================================
# 12. PERSONAL BASELINE (ADVANCED)
# ================================
baseline = X_train.mean()

df['deviation'] = abs(X - baseline).sum(axis=1)

threshold = df['deviation'].quantile(0.90)
df['alert'] = df['deviation'] > threshold

print("\nBaseline anomaly detection added!")


# ================================
# 13. FINAL OUTPUT SAMPLE
# ================================
print("\n===== FINAL DATA SAMPLE =====")
print(df[['time', 'hr_mean', 'eda_mean', 'acc_magnitude', 'deviation', 'alert']].head())


print("\n✅ PIPELINE COMPLETED SUCCESSFULLY!")