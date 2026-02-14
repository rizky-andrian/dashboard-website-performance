# ==============================================
# DASHBOARD WEBSITE PERFORMANCES
# ==============================================
# Framework: Streamlit + Plotly + Scikit-learn
# ==============================================

import streamlit as st
import pandas as pd
import numpy as np
import glob, os
from io import BytesIO
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# Konfigurasi Tampilan
# ---------------------------where
st.set_page_config(
    page_title="Dashboard Website Performance",
    page_icon="💻",
    layout="wide"
)

# Warna tema
PRIMARY_COLOR = "#2E86C1"
SECONDARY_COLOR = "#1ABC9C"
BACKGROUND = "#F7F9FB"

st.markdown(
    f"""
    <style>
        .main {{
            background-color: {BACKGROUND};
        }}
        h1, h2, h3, h4, h5 {{
            color: {PRIMARY_COLOR};
        }}
        .stDataFrame {{
            border-radius: 12px;
            border: 1px solid #ddd;
            background-color: white;
        }}
        div[data-testid="stMetricValue"] {{
            color: {SECONDARY_COLOR};
        }}
        .stDownloadButton > button {{
            background-color: {PRIMARY_COLOR};
            color: white;
            border-radius: 8px;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# Fungsi bantu
# ---------------------------
def find_and_concat(pattern):
    dfs = []
    for p in glob.glob(pattern):
        df = pd.read_csv(p)
        df["_source"] = os.path.basename(p)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def to_excel_bytes(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name='Clusters')
    return output.getvalue()

def style_card(text, value, color):
    st.markdown(
        f"""
        <div style='padding:15px; border-radius:10px; background-color:{color}; color:white;'>
        <h4 style='margin-bottom:5px;'>{text}</h4>
        <h2 style='margin-top:0px;'>{value}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.title("⚙️ Pengaturan Dashboard")
st.sidebar.write("Upload file hasil web scraping:")
res_files = st.sidebar.file_uploader("resource_mining (*.csv)", type="csv", accept_multiple_files=True)
resp_files = st.sidebar.file_uploader("response_times_mining (*.csv)", type="csv", accept_multiple_files=True)
use_minmax = st.sidebar.checkbox("Gunakan MinMaxScaler", True)
n_cluster = st.sidebar.slider("Jumlah Cluster (K)", 2, 6, 3)

# ---------------------------
# Load Data
# ---------------------------
st.title("💻 Dashboard Website Performance")

if res_files:
    resource_df = pd.concat([pd.read_csv(f) for f in res_files])
else:
    resource_df = find_and_concat("resource_mining*.csv")

if resp_files:
    response_df = pd.concat([pd.read_csv(f) for f in resp_files])
else:
    response_df = find_and_concat("response_times_mining*.csv")

if resource_df.empty and response_df.empty:
    st.warning("⚠️ Belum ada data. Upload file CSV terlebih dahulu di sidebar.")
    st.stop()

# ---------------------------
# Data Cleaning
# ---------------------------
def clean_resource(df):
    for c in ["trial", "resource_url", "status"]:
        if c in df.columns: df.drop(columns=c, inplace=True)
    df.dropna(inplace=True)
    return df

def clean_response(df):
    for c in ["trial", "status"]:
        if c in df.columns: df.drop(columns=c, inplace=True)
    df.dropna(inplace=True)
    return df

resource_df = clean_resource(resource_df)
response_df = clean_response(response_df)

# ---------------------------
# Tampilan Data
# ---------------------------
st.header("📊 Data Awal")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Resource Mining")
    st.dataframe(resource_df.head(8))
    st.caption(f"{len(resource_df)} baris data")

with col2:
    st.subheader("Response Times Mining")
    st.dataframe(response_df.head(8))
    st.caption(f"{len(response_df)} baris data")

# ---------------------------
# Statistik & Visualisasi
# ---------------------------
st.header("📈 Analisis Performa Website")

colA, colB, colC = st.columns(3)
if "page_load_time_s" in resource_df.columns:
    avg_load = resource_df["page_load_time_s"].mean()
    style_card("Rata-rata Load Time", f"{avg_load:.2f} s", PRIMARY_COLOR)

if "response_time_s" in response_df.columns:
    avg_resp = response_df["response_time_s"].mean()
    style_card("Rata-rata Response Time", f"{avg_resp:.2f} s", SECONDARY_COLOR)

with colC:
    if "url" in response_df.columns:
        site_count = response_df["url"].nunique()
        style_card("Jumlah Website Dianalisis", str(site_count), "#45B39D")

st.subheader("🔹 Hubungan Ukuran Halaman vs Response Time")
if "page_size_bytes" in response_df.columns and "response_time_s" in response_df.columns:
    fig = px.scatter(
        response_df, x="page_size_bytes", y="response_time_s",
        color_discrete_sequence=[PRIMARY_COLOR],
        hover_data=["url"] if "url" in response_df.columns else None,
        title="Hubungan Ukuran Halaman & Waktu Respons",
        labels={"page_size_bytes": "Ukuran Halaman (Bytes)", "response_time_s": "Response Time (s)"}
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("🔹 Jumlah Elemen UX/UI per Website")
elems = [c for c in ["n_img", "n_js", "n_css", "n_button"] if c in response_df.columns]
if elems:
    agg = response_df.groupby("url")[elems].mean().reset_index()
    melt = agg.melt(id_vars="url", var_name="Elemen", value_name="Rata-rata")
    fig2 = px.bar(melt, x="url", y="Rata-rata", color="Elemen", barmode="group",
                  color_discrete_sequence=px.colors.qualitative.Safe)
    st.plotly_chart(fig2, use_container_width=True)

# ---------------------------
# Clustering
# ---------------------------
st.header("🤖 K-Means Clustering")
numeric_cols = response_df.select_dtypes(include=np.number).columns.tolist()
sel_cols = st.multiselect("Pilih fitur untuk clustering", numeric_cols, default=numeric_cols[:3])
if len(sel_cols) < 2:
    st.warning("Pilih minimal dua kolom numerik untuk clustering.")
    st.stop()

X = response_df[sel_cols].dropna()
scaler = MinMaxScaler() if use_minmax else StandardScaler()
X_scaled = scaler.fit_transform(X)

model = KMeans(n_clusters=n_cluster, random_state=42, n_init=20)
labels = model.fit_predict(X_scaled)

response_df["Cluster"] = labels
fig3 = px.scatter(
    response_df, x=sel_cols[0], y=sel_cols[1],
    color=response_df["Cluster"].astype(str),
    color_discrete_sequence=px.colors.qualitative.Bold,
    hover_data=["url"] if "url" in response_df.columns else None,
    title="Visualisasi Clustering Website"
)
st.plotly_chart(fig3, use_container_width=True)

st.subheader("📋 Rata-rata Tiap Cluster")
summary = response_df.groupby("Cluster")[sel_cols].mean().round(3)
summary["Jumlah Data"] = response_df.groupby("Cluster").size()
st.dataframe(summary.style.background_gradient(cmap="Blues"))

# ---------------------------
# Unduh hasil
# ---------------------------
excel_data = to_excel_bytes(response_df)
st.download_button(
    label="💾 Download Hasil Cluster (.xlsx)",
    data=excel_data,
    file_name="hasil_cluster.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# ---------------------------
# Insight otomatis
# ---------------------------
st.header("💡 Insight & Rekomendasi")
insight = []
if avg_load > 1.5:
    insight.append("⏱️ Waktu muat cukup tinggi, optimalkan kompresi gambar & minify CSS/JS.")
else:
    insight.append("🚀 Performa loading sudah efisien (<1.5s).")

if avg_resp > 1.0:
    insight.append("🌐 Waktu respon server agak tinggi, pertimbangkan caching & CDN.")
else:
    insight.append("🟢 Server memiliki waktu respon cepat.")

best_cluster = summary[sel_cols[0]].idxmin()
insight.append(f"🏆 Cluster {best_cluster} menunjukkan performa paling efisien.")

for i in insight:
    st.markdown(f"- {i}")

st.caption("📘 Dibuat berdasarkan penelitian 'Optimasi Performa Website & UX/UI Berbasis Data Mining dan Web Scraping' (2025)")
