from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from generate_data import make_data

BASE = Path(__file__).parent
st.set_page_config(page_title="Predictive Maintenance", page_icon="⚙️", layout="wide")
st.title("⚙️ Predictive Maintenance & Failure Risk")
st.caption("Portfolio demo using synthetic sensor data. Not for safety-critical decisions.")
model_path = BASE / "model.joblib"
if not model_path.exists():
    st.warning("Model not trained yet. Run `python train.py` in this project folder, then refresh.")
    st.stop()
bundle = joblib.load(model_path)
model, features = bundle["model"], bundle["features"]
c1, c2, c3, c4 = st.columns(4)
temp = c1.number_input("Temperature (°C)", 35.0, 115.0, 78.0)
vib = c2.number_input("Vibration (mm/s)", 0.1, 12.0, 3.0)
pressure = c3.number_input("Pressure (kPa)", 55.0, 145.0, 100.0)
runtime = c4.number_input("Runtime hours", 100, 12000, 6500)
if st.button("Estimate failure risk", type="primary"):
    row = pd.DataFrame([[temp, vib, pressure, runtime]], columns=features)
    risk = float(model.predict_proba(row)[0, 1])
    st.metric("Estimated failure risk", f"{risk:.1%}")
    if risk >= 0.65:
        st.error("Higher estimated risk — inspect the equipment and validate with an engineer.")
    elif risk >= 0.35:
        st.warning("Moderate estimated risk — review sensor trends.")
    else:
        st.success("Lower estimated risk in this demo model.")
st.subheader("Training data overview")
df = pd.read_csv(BASE / "data" / "sensor_readings.csv") if (BASE / "data" / "sensor_readings.csv").exists() else make_data()
a, b = st.columns(2)
with a:
    st.write("Failure label distribution")
    counts = df["failure_next_24h"].value_counts().sort_index()
    fig, ax = plt.subplots()
    ax.bar(["No failure", "Failure"], [counts.get(0, 0), counts.get(1, 0)])
    ax.set_ylabel("Rows")
    st.pyplot(fig)
with b:
    st.write("Sensor relationships")
    fig, ax = plt.subplots()
    ax.scatter(df["temperature_c"], df["vibration_mm_s"], c=df["failure_next_24h"], alpha=0.35)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Vibration (mm/s)")
    st.pyplot(fig)
metrics_path = BASE / "metrics.json"
if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text())
    st.subheader("Held-out test metrics")
    cols = st.columns(5)
    for col, key in zip(cols, ["accuracy", "precision", "recall", "f1", "roc_auc"]):
        col.metric(key.upper(), f"{metrics[key]:.3f}")
