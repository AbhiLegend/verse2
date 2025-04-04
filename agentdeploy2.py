import streamlit as st
import pandas as pd
import os
import json
from glob import glob

st.set_page_config(page_title="💊 Drug Discovery Dashboard", layout="wide")

st.title("🧬 Drug Discovery Dashboard")
st.caption("Visualize AI-generated molecule candidates with filters, stats, and charts.")

# 🕒 Auto-refresh every 10s
st.markdown(
    """
    <meta http-equiv="refresh" content="10">
    """,
    unsafe_allow_html=True
)

# 📁 Find latest job
result_dirs = sorted(glob("results/job_*"), reverse=True)
if not result_dirs:
    st.warning("No results available. Please run `main_agents.py` first.")
    st.stop()

latest_dir = result_dirs[0]
job_id = os.path.basename(latest_dir)
json_path = os.path.join(latest_dir, "top_candidates.json")
csv_path = os.path.join(latest_dir, "top_candidates.csv")

st.markdown(f"### 🔬 Latest Job ID: `{job_id}`")

# 📄 Load Data
try:
    with open(json_path, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
except Exception as e:
    st.error(f"Error loading JSON: {e}")
    st.stop()

if df.empty:
    st.warning("No molecule data available.")
    st.stop()

# 🎛️ Sidebar Filters
st.sidebar.header("🔎 Filter Molecules")
logp_range = st.sidebar.slider("LogP", 0.0, 6.0, (0.0, 3.0))
mw_range = st.sidebar.slider("Molecular Weight", 0, 600, (0, 500))
toxicity = st.sidebar.selectbox("Toxicity", ["All"] + sorted(df["toxicity"].unique()))

filtered = df[
    (df["logp"] >= logp_range[0]) & (df["logp"] <= logp_range[1]) &
    (df["mw"] >= mw_range[0]) & (df["mw"] <= mw_range[1])
]
if toxicity != "All":
    filtered = filtered[filtered["toxicity"] == toxicity]

# 🧠 Stats
st.subheader("📊 Molecule Stats")
col1, col2, col3 = st.columns(3)
col1.metric("Total", len(filtered))
col2.metric("Avg Affinity", f"{filtered['affinity_score'].mean():.2f}" if not filtered.empty else "–")
col3.metric("Low-Risk", len(filtered[filtered["toxicity"] == "Low Risk"]))

# 📈 Property Charts
st.markdown("### 📈 Molecular Properties")
chart1, chart2 = st.columns(2)
with chart1:
    st.bar_chart(filtered["logp"])
with chart2:
    st.bar_chart(filtered["mw"])

# 🧬 Molecule Grid
st.markdown("### 🧬 Molecule Images")
cols = st.columns(3)
for i, row in enumerate(filtered.itertuples()):
    with cols[i % 3]:
        st.image(row.image_path, caption=f"{row.smiles}\nAff: {row.affinity_score} | {row.toxicity}", use_column_width=True)

# 📥 Download
st.markdown("### 📥 Export Data")
col_dl1, col_dl2 = st.columns(2)
with open(csv_path, "rb") as f:
    col_dl1.download_button("⬇️ Download CSV", f, file_name=f"{job_id}.csv")
with open(json_path, "rb") as f:
    col_dl2.download_button("⬇️ Download JSON", f, file_name=f"{job_id}.json")
