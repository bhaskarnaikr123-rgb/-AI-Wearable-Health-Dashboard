import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import zipfile
import tempfile
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(layout="wide", page_title="AI Health Dashboard")

st.title("🧠 AI Wearable Health Dashboard")

# =========================================
# SIDEBAR
# =========================================
uploaded_zip = st.sidebar.file_uploader("📂 Upload Dataset (ZIP)", type="zip")

# =========================================
# LOAD FUNCTION
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

    freq_ms = int((1 / sampling_rate) * 1000)

    df['time'] = pd.date_range(
        start=pd.to_datetime(start_time, unit='s'),
        periods=len(df),
        freq=f"{freq_ms}ms"
    )

    return df

# =========================================
# FIND FILE IN ZIP
# =========================================
def find_file(root, filename):
    for path, dirs, files in os.walk(root):
        if filename in files:
            return os.path.join(path, filename)
    return None

# =========================================
# PROCESS ZIP
# =========================================
def process_zip(uploaded_zip):
    temp_dir = tempfile.mkdtemp()

    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    eda_path = find_file(temp_dir, "EDA.csv")
    temp_path = find_file(temp_dir, "TEMP.csv")
    hr_path = find_file(temp_dir, "HR.csv")
    acc_path = find_file(temp_dir, "ACC.csv")

    if not all([eda_path, temp_path, hr_path, acc_path]):
        st.error("❌ Missing files inside ZIP")
        st.stop()

    eda = load_signal(eda_path, "EDA")
    temp = load_signal(temp_path, "TEMP")
    hr = load_signal(hr_path, "HR")
    acc = load_signal(acc_path, ['x','y','z'])

    df = eda.merge(temp, on='time')
    df = df.merge(hr, on='time')
    df = df.merge(acc, on='time')

    return df.sort_values(by='time').reset_index(drop=True)

# =========================================
# DEFAULT DATA (IF NO ZIP)
# =========================================
def load_default():
    eda = load_signal("EDA.csv", "EDA")
    temp = load_signal("TEMP.csv", "TEMP")
    hr = load_signal("HR.csv", "HR")
    acc = load_signal("ACC.csv", ['x','y','z'])

    df = eda.merge(temp, on='time')
    df = df.merge(hr, on='time')
    df = df.merge(acc, on='time')

    return df.sort_values(by='time').reset_index(drop=True)

# =========================================
# LOAD DATA
# =========================================
if uploaded_zip is not None:
    st.sidebar.success("ZIP uploaded ✅")
    df = process_zip(uploaded_zip)
else:
    st.sidebar.info("Using default data")
    df = load_default()

# =========================================
# FEATURES
# =========================================
df['acc_magnitude'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
df['hr_mean'] = df['HR'].rolling(10).mean()
df['eda_mean'] = df['EDA'].rolling(10).mean()
df['hrv'] = df['HR'].rolling(10).std()

df = df.dropna()

# LABEL
df['label'] = 0
df.loc[
    (df['eda_mean'] > df['eda_mean'].quantile(0.75)) &
    (df['hr_mean'] > df['hr_mean'].quantile(0.60)),
    'label'
] = 1

# =========================================
# MODEL
# =========================================
features = ['acc_magnitude','hr_mean','eda_mean','hrv']
X = df[features]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train,y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))

# =========================================
# KPI
# =========================================
c1, c2, c3 = st.columns(3)
c1.metric("Accuracy", f"{accuracy*100:.1f}%")
c2.metric("Data Points", len(df))
c3.metric("Stress Events", int(df['label'].sum()))

st.divider()

# =========================================
# CHARTS
# =========================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("Heart Rate")
    st.plotly_chart(px.line(df, x='time', y='HR'), use_container_width=True)

with col2:
    st.subheader("EDA")
    st.plotly_chart(px.line(df, x='time', y='EDA'), use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    st.subheader("Activity")
    st.plotly_chart(px.line(df, x='time', y='acc_magnitude'), use_container_width=True)

with col4:
    st.subheader("HRV")
    st.plotly_chart(px.line(df, x='time', y='hrv'), use_container_width=True)

st.divider()

# =========================================
# STRESS
# =========================================
st.subheader("🧠 Stress Detection")
st.plotly_chart(px.scatter(df, x='time', y='hr_mean', color='label'), use_container_width=True)

# =========================================
# ALERTS
# =========================================
st.subheader("⚠️ Alerts")

baseline = X_train.mean()
df['deviation'] = abs(X - baseline).sum(axis=1)

st.plotly_chart(px.line(df, x='time', y='deviation'), use_container_width=True)

st.success("🚀 Dashboard Ready")