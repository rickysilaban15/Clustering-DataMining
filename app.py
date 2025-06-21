import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import plotly.express as px
import io
import base64
from fpdf import FPDF

# 🌸 Custom Wine-Themed Background
page_bg_img = '''
<style>
[data-testid="stAppViewContainer"] {
background-image: url("https://images.unsplash.com/photo-1608417639543-1e4e1ea98e8f");
background-size: cover;
background-attachment: fixed;
}
[data-testid="stSidebar"] {
background-color: rgba(60, 30, 20, 0.95);
color: white;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# 🎯 Title and Sidebar
st.title("🍷 WineCluster Pro - Analisis Clustering Wine")
st.sidebar.title("🍇 WineCluster Sidebar")

# 📤 Upload File
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"], key="upload_1") 

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

    # 📌 Pilih Kolom Numerik
    cols = st.multiselect(
        "Pilih Kolom Numerik untuk Clustering",
        df.select_dtypes(include=np.number).columns.tolist(),
        default=df.select_dtypes(include=np.number).columns.tolist()
    )
    if len(cols) < 2:
        st.warning("Pilih minimal 2 kolom untuk clustering.")
        st.stop()

    # ✨ Standardisasi
    X = StandardScaler().fit_transform(df[cols])

    # ⚙️ Metode Clustering
    method = st.sidebar.selectbox("Metode Clustering", ["KMeans", "DBSCAN", "Agglomerative", "GMM"])

    if method == "KMeans":
        k = st.sidebar.slider("Jumlah Klaster (K)", 2, 10, 3)
        model = KMeans(n_clusters=k, random_state=42)
        cluster_labels = model.fit_predict(X)
        score = silhouette_score(X, cluster_labels)
        st.success(f"Silhouette Score: {score:.3f}")

    elif method == "DBSCAN":
        eps = st.sidebar.slider("EPS (radius tetangga)", 0.1, 5.0, 0.5)
        min_samples = st.sidebar.slider("Min Samples", 2, 10, 5)
        model = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = model.fit_predict(X)

    elif method == "Agglomerative":
        k = st.sidebar.slider("Jumlah Klaster (K)", 2, 10, 3)
        model = AgglomerativeClustering(n_clusters=k)
        cluster_labels = model.fit_predict(X)

    elif method == "GMM":
        k = st.sidebar.slider("Jumlah Klaster (K)", 2, 10, 3)
        model = GaussianMixture(n_components=k, random_state=42)
        cluster_labels = model.fit_predict(X)

    df['Cluster'] = cluster_labels

    # 📊 Visualisasi Scatter 2D
    st.subheader("📊 Visualisasi Klaster (2D)")
    fig = px.scatter(
        df,
        x=cols[0],
        y=cols[1],
        color=df['Cluster'].astype(str),
        symbol=df['Cluster'].astype(str),
        title="Scatterplot Clustering"
    )
    st.plotly_chart(fig)

    # 📌 Statistik per Cluster
    st.subheader("📌 Statistik per Cluster")
    st.dataframe(df.groupby('Cluster')[cols].mean().round(2))

    # 📈 Distribusi Pie Chart
    st.subheader("📈 Distribusi Jumlah Data Tiap Cluster")
    fig2 = px.pie(df, names='Cluster', title="Distribusi Data per Cluster")
    st.plotly_chart(fig2)

    # 💾 Download ke CSV
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button("💾 Download Hasil Clustering (.CSV)", data=csv_data, file_name="hasil_clustering.csv", mime='text/csv')

    # 📥 Download ke Excel
    excel_buffer = io.BytesIO()
    df.to_excel(excel_buffer, index=False)
    st.download_button("📥 Download ke Excel (.XLSX)", data=excel_buffer, file_name="hasil_clustering.xlsx")

    # 📄 Export PDF Ringkasan
    def export_pdf(df):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Laporan Analisis Clustering Wine", ln=True, align="C")
        pdf.ln(10)
        pdf.multi_cell(0, 10, txt="Ringkasan Statistik per Klaster:")

        cluster_summary = df.groupby('Cluster')[cols].mean().round(2)

        for cluster in cluster_summary.index:
            pdf.ln(5)
            pdf.cell(0, 10, f"Cluster {cluster}", ln=True)
            for col in cols:
                pdf.cell(0, 10, f"{col}: {cluster_summary.loc[cluster, col]}", ln=True)

        # Perbaikan: kembalikan bytearray sebagai bytes
        return bytes(pdf.output(dest='S'))



    pdf_data = export_pdf(df)
    st.download_button("📄 Export Laporan PDF", data=pdf_data, file_name="laporan_wine_clustering.pdf")

    # 📌 Footer
    st.markdown("---")
    st.caption("🍷 WineCluster Pro by Ricky Steven Silaban - T/A Data Mining 2025")
else:
    st.info("Silakan upload file CSV terlebih dahulu.")

