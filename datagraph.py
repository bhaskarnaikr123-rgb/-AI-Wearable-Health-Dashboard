# =========================================
# WEARABLE AI PROJECT (FULL PIPELINE + GRAPHS)
# =========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

plt.style.use('ggplot')

# =========================================
# 1. LOAD FUNCTION (FIXED VERSION)
# =========================================
def load_signal(file_path, col_names):
    df = pd.read_csv(file_path, header=None)

    start_time = df.iloc[0, 0]
    sampling_rate = df.iloc[1, 0]

    df = df.iloc[2:].reset_index(drop=True)

    if isinstance(col_names, list):
        df.columns = col_names
    else:
        df.columns = [col_names]

    # Convert to milliseconds (FIXED)
    freq_ms = int((1 / sampling_rate) * 1000)

    df['time'] = pd.date_range(
        start=pd.to_datetime(start_time, unit='s'),
        periods=len(df),
        freq=f"{freq_ms}ms"
    )

    return df


# =========================================
# 2. LOAD DATA
# =========================================
eda = load_signal("EDA.csv", "EDA")
temp = load_signal("TEMP.csv", "TEMP")
hr = load_signal("HR.csv", "HR")
acc = load_signal("ACC.csv", ['x', 'y', 'z'])

print("✅ Data Loaded")


# =========================================
# 3. MERGE DATA
# =========================================
df = eda.merge(temp, on='time')
df = df.merge(hr, on='time')
df = df.merge(acc, on='time')

df = df.sort_values(by='time').reset_index(drop=True)

print("✅ Data Merged")


# =========================================
# 4. FEATURE ENGINEERING
# =========================================
df['acc_magnitude'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
df['activity_mean'] = df['acc_magnitude'].rolling(50).mean()

df['hr_mean'] = df['HR'].rolling(10).mean()
df['hr_std'] = df['HR'].rolling(10).std()

df['hrv'] = df['HR'].rolling(10).std()

df['eda_mean'] = df['EDA'].rolling(10).mean()
df['eda_std'] = df['EDA'].rolling(10).std()

df['temp_mean'] = df['TEMP'].rolling(50).mean()

print("✅ Features Created")


# =========================================
# 5. LABEL CREATION (TEMPORARY)
# =========================================
df['label'] = 0

df.loc[
    (df['eda_mean'] > df['eda_mean'].quantile(0.75)) &
    (df['hr_mean'] > df['hr_mean'].quantile(0.60)),
    'label'
] = 1

print("✅ Labels Created")


# =========================================
# 6. CLEAN DATA
# =========================================
df = df.dropna().reset_index(drop=True)


# =========================================
# 7. FEATURE SELECTION
# =========================================
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


# =========================================
# 8. TRAIN TEST SPLIT
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =========================================
# 9. MODEL TRAINING
# =========================================
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

print("✅ Model Trained")


# =========================================
# 10. EVALUATION
# =========================================
y_pred = model.predict(X_test)

print("\n===== MODEL PERFORMANCE =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# =========================================
# 11. RISK SCORE
# =========================================
sample = X_test.iloc[0:1]
print("\nPrediction:", model.predict(sample))
print("Risk Score:", model.predict_proba(sample))


# =========================================
# 12. BASELINE ANOMALY DETECTION
# =========================================
baseline = X_train.mean()
df['deviation'] = abs(X - baseline).sum(axis=1)

threshold = df['deviation'].quantile(0.90)
df['alert'] = df['deviation'] > threshold

print("✅ Anomaly Detection Added")


# =========================================
# 13. VISUALIZATIONS
# =========================================

# ---- HR Graph ----
plt.figure(figsize=(12,5))
plt.plot(df['time'], df['HR'])
plt.title("Heart Rate Over Time")
plt.xticks(rotation=45)
plt.show()

# ---- Multi Signal ----
plt.figure(figsize=(12,6))
plt.plot(df['time'], df['HR'], label='HR')
plt.plot(df['time'], df['EDA'], label='EDA')
plt.plot(df['time'], df['TEMP'], label='TEMP')
plt.legend()
plt.title("Signals Comparison")
plt.xticks(rotation=45)
plt.show()

# ---- Activity ----
plt.figure(figsize=(12,5))
plt.plot(df['time'], df['acc_magnitude'])
plt.title("Activity Level")
plt.xticks(rotation=45)
plt.show()

# ---- Stress Visualization ----
plt.figure(figsize=(12,5))
plt.plot(df['time'], df['hr_mean'], label='HR')
plt.scatter(df['time'], df['label']*df['hr_mean'], color='red', label='Stress')
plt.legend()
plt.title("Stress Detection")
plt.xticks(rotation=45)
plt.show()

# ---- Correlation ----
plt.figure(figsize=(8,6))
sns.heatmap(df[features].corr(), annot=True)
plt.title("Feature Correlation")
plt.show()

# ---- Feature Importance ----
importance = model.feature_importances_
feat_imp = pd.Series(importance, index=features)

feat_imp.sort_values().plot(kind='barh')
plt.title("Feature Importance")
plt.show()

# ---- Anomaly Detection ----
plt.figure(figsize=(12,5))
plt.plot(df['time'], df['deviation'], label='Deviation')
plt.scatter(df[df['alert']]['time'],
            df[df['alert']]['deviation'],
            color='red', label='Alert')
plt.legend()
plt.title("Anomaly Detection")
plt.xticks(rotation=45)
plt.show()


print("\n🚀 FULL PIPELINE COMPLETED SUCCESSFULLY!")