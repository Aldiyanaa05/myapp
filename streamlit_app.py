# thermal_oxidation.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Thermal Oxidation Simulation App")

# Sample CSV generator (so you can test immediately)
sample_df = pd.DataFrame({
    "time": [0,10,20,30,40,50],
    "A": [0.1]*6,
    "B0": [0.01]*6,
    "Ea": [1.2]*6,
    "T": [1200]*6,
    "alpha": [0.001]*6,
    "reliability_dry": [0.98,0.97,0.96,0.95,0.94,0.93],
    "reliability_wet": [0.97,0.96,0.95,0.94,0.93,0.92],
    "quantum_factor": [1.0]*6
})
csv_bytes = sample_df.to_csv(index=False).encode("utf-8")
st.sidebar.download_button("Download sample CSV (for testing)", data=csv_bytes, file_name="oxidation_data.csv", mime="text/csv")

uploaded_file = st.file_uploader("Upload your oxide growth CSV", type=["csv"])

# Sidebar parameters (defaults taken from uploaded file if provided)
st.sidebar.header("Manual Input (overrides CSV first-row if changed)")
A = st.sidebar.number_input("A", value=0.1, format="%.6f")
B0 = st.sidebar.number_input("B0", value=0.01, format="%.6f")
Ea = st.sidebar.number_input("Ea (eV)", value=1.2, format="%.4f")
T = st.sidebar.number_input("Temperature (K)", value=1200.0, format="%.1f")
alpha = st.sidebar.number_input("alpha", value=0.001, format="%.6f")
wet_scale = st.sidebar.slider("Wet scale factor", 1.0, 3.0, 1.5, step=0.1)

if uploaded_file is None:
    st.info("No CSV uploaded — use the sample CSV (download from sidebar) or upload your own.")
    st.stop()

# Load CSV
data = pd.read_csv(uploaded_file)

# Ensure numeric & sorted by time
data['time'] = pd.to_numeric(data['time'], errors='coerce')
data = data.sort_values('time').reset_index(drop=True)

# If file contains parameter columns, use first row as defaults unless user changed them
A = float(data['A'][0]) if 'A' in data.columns else A
B0 = float(data['B0'][0]) if 'B0' in data.columns else B0
Ea = float(data['Ea'][0]) if 'Ea' in data.columns else Ea
T = float(data['T'][0]) if 'T' in data.columns else T
alpha = float(data['alpha'][0]) if 'alpha' in data.columns else alpha

k_B = 8.617e-5  # eV/K
time = data['time'].to_numpy(dtype=float)

# Calculations
thickness_dry = np.sqrt(B0 * time + (alpha * time)**2) * np.exp(-Ea / (k_B * T))
thickness_wet = wet_scale * np.sqrt(B0 * time + (alpha * time)**2) * np.exp(-Ea / (k_B * T))

growth_rate_dry = np.gradient(thickness_dry, time, edge_order=2)
growth_rate_wet = np.gradient(thickness_wet, time, edge_order=2)

reliability_dry = data['reliability_dry'] if 'reliability_dry' in data.columns else np.ones_like(time)
reliability_wet = data['reliability_wet'] if 'reliability_wet' in data.columns else np.ones_like(time)
quantum = data['quantum_factor'] if 'quantum_factor' in data.columns else np.ones_like(time)

quality_dry = thickness_dry * quantum
quality_wet = thickness_wet * (quantum * 0.95)

# Show parameter summary
st.markdown("**Used parameters:**")
st.write(dict(A=A, B0=B0, Ea=Ea, T=T, alpha=alpha, wet_scale=wet_scale))

# Plots: create a 2x2 layout
fig, axs = plt.subplots(2,2, figsize=(12,8))
axs = axs.flatten()

axs[0].plot(time, thickness_dry, label="Dry Oxide")
axs[0].plot(time, thickness_wet, label="Wet Oxide")
axs[0].set_title("Oxide Thickness (nm)")
axs[0].set_xlabel("Time (s)")
axs[0].legend()
axs[0].grid(True)

axs[1].plot(time, growth_rate_dry, '--', label="Dry Growth Rate")
axs[1].plot(time, growth_rate_wet, '--', label="Wet Growth Rate")
axs[1].set_title("Growth Rate (nm/s)")
axs[1].set_xlabel("Time (s)")
axs[1].legend()
axs[1].grid(True)

axs[2].plot(time, reliability_dry, label="Dry Reliability")
axs[2].plot(time, reliability_wet, label="Wet Reliability")
axs[2].set_title("Reliability")
axs[2].set_xlabel("Time (s)")
axs[2].set_ylim(0.0, 1.05)
axs[2].legend()
axs[2].grid(True)

axs[3].plot(time, quality_dry, label="Dry Quality")
axs[3].plot(time, quality_wet, label="Wet Quality")
axs[3].set_title("Quality (Quantum Effects)")
axs[3].set_xlabel("Time (s)")
axs[3].legend()
axs[3].grid(True)

st.pyplot(fig)

# Prepare results CSV for download
results_df = pd.DataFrame({
    "time": time,
    "thickness_dry": thickness_dry,
