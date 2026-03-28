import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(layout="wide", page_title="AI Health Dashboard")

# =========================================
# DARK THEME STYLE
# =========================================
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: white;
}
.metric-card {
    background: #1c1f26;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.title("🧠 AI Wearable Health Dashboard")

# =========================================
# SIDEBAR (CONTROL PANEL)
# =========================================
st.sidebar.header("⚙️ Controls")

file_uploaded = st.sidebar.file_uploader("Upload CSV (optional)")

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
# LOAD DATA
# =========================================
eda = load_signal("EDA.csv", "EDA")
temp = load_signal("TEMP.csv", "TEMP")
hr = load_signal("HR.csv", "HR")
acc = load_signal("ACC.csv", ['x','y','z'])

df = eda.merge(temp, on='time')
df = df.merge(hr, on='time')
df = df.merge(acc, on='time')

df = df.sort_values(by='time').reset_index(drop=True)

# =========================================
# FEATURES
# =========================================
df['acc_magnitude'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
df['hr_mean'] = df['HR'].rolling(10).mean()
df['eda_mean'] = df['EDA'].rolling(10).mean()
df['hrv'] = df['HR'].rolling(10).std()

df['label'] = 0
df.loc[
    (df['eda_mean'] > df['eda_mean'].quantile(0.75)) &
    (df['hr_mean'] > df['hr_mean'].quantile(0.60)),
    'label'
] = 1

df = df.dropna()

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
# KPI CARDS
# =========================================
col1, col2, col3 = st.columns(3)

col1.markdown(f"""
<div class="metric-card">
<h3>Accuracy</h3>
<h2>{accuracy*100:.1f}%</h2>
</div>
""", unsafe_allow_html=True)

col2.markdown(f"""
<div class="metric-card">
<h3>Data Points</h3>
<h2>{len(df)}</h2>
</div>
""", unsafe_allow_html=True)

col3.markdown(f"""
<div class="metric-card">
<h3>Stress Events</h3>
<h2>{int(df['label'].sum())}</h2>
</div>
""", unsafe_allow_html=True)

st.divider()

# =========================================
# INTERACTIVE CHARTS
# =========================================

col4, col5 = st.columns(2)

with col4:
    st.subheader("Heart Rate")
    fig = px.line(df, x='time', y='HR')
    st.plotly_chart(fig, use_container_width=True)

with col5:
    st.subheader("EDA")
    fig = px.line(df, x='time', y='EDA')
    st.plotly_chart(fig, use_container_width=True)

col6, col7 = st.columns(2)

with col6:
    st.subheader("Activity")
    fig = px.line(df, x='time', y='acc_magnitude')
    st.plotly_chart(fig, use_container_width=True)

with col7:
    st.subheader("HRV")
    fig = px.line(df, x='time', y='hrv')
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# =========================================
# STRESS DETECTION
# =========================================
st.subheader("🧠 Stress Detection")

fig = px.scatter(df, x='time', y='hr_mean', color='label')
st.plotly_chart(fig, use_container_width=True)

# =========================================
# ANOMALY DETECTION
# =========================================
st.subheader("⚠️ Alerts")

baseline = X_train.mean()
df['deviation'] = abs(X - baseline).sum(axis=1)

fig = px.line(df, x='time', y='deviation')
st.plotly_chart(fig, use_container_width=True)

st.success("🚀 Premium Dashboard Running")