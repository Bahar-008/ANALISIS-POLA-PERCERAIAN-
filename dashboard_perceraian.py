import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Set konfigurasi halaman Streamlit
st.set_page_config(page_title="Analisis Kasus Perceraian", layout="wide")

# Set gaya visualisasi
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Judul Aplikasi
st.title("📊 Aplikasi Analisis & Pola Kasus Perceraian")
st.markdown("Aplikasi ini menganalisis tren, faktor dominan, distribusi usia, serta pengelompokan pola perceraian.")
st.markdown("---")

# Sidebar untuk Unggah File
st.sidebar.header("Pengaturan Data")
uploaded_file = st.sidebar.file_uploader("Unggah File CSV Perkara Cerai", type=["csv"])

@st.cache_data
def load_and_clean_data(file):
    # Membaca data dengan delimiter semicolon (;)
    df = pd.read_csv(file, sep=';', encoding='latin-1')
    
    # Konversi tipe data numeric
    df['umur_p'] = pd.to_numeric(df['umur_p'], errors='coerce')
    df['umur_t'] = pd.to_numeric(df['umur_t'], errors='coerce')
    df['lama_pernikahan_bulan'] = pd.to_numeric(df['lama_pernikahan_bulan'], errors='coerce')
    
    # Konversi tanggal
    df['tanggal_putusan'] = pd.to_datetime(df['tanggal_putusan'], errors='coerce')
    
    # Drop data yang esensialnya kosong untuk analisis numerik
    df = df.dropna(subset=['umur_p', 'umur_t'])
    return df

# Memastikan file sudah diunggah sebelum menjalankan analisis
if uploaded_file is not None:
    try:
        # 1. Memuat Data
        df = load_and_clean_data(uploaded_file)
        st.success(f"🎉 Data berhasil dimuat! Total baris siap analisis: **{len(df)}**")
        
        # Pembuatan Tabs agar tampilan rapi dan interaktif
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Ringkasan & Tren", 
            "💡 Faktor & Usia", 
            "🤖 Pola Clustering",
            "📋 Data Mentah"
        ])

        # ====================================================================
        # TAB 1: RINGKASAN & TREN (Statistik Deskriptif & Time Series)
        # ====================================================================
        with tab1:
            st.header("Ringkasan Data & Tren Waktu")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Statistik Deskriptif")
                st.dataframe(df[['umur_p', 'umur_t', 'lama_pernikahan_bulan']].describe())
            
            with col2:
                st.subheader("Tren Kasus Perceraian")
                df_ts = df.dropna(subset=['tanggal_putusan']).copy()
                
                if not df_ts.empty:
                    df_ts['tahun_bulan'] = df_ts['tanggal_putusan'].dt.to_period('M')
                    trend_bulanan = df_ts.groupby('tahun_bulan').size()
                    
                    # Konversi index period ke string agar tidak error saat di-plot Streamlit/Matplotlib
                    trend_bulanan.index = trend_bulanan.index.astype(str)
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    trend_bulanan.plot(kind='line', marker='o', color='red', linewidth=2, ax=ax)
                    ax.set_title('Tren Jumlah Kasus Perceraian Berdasarkan Tanggal Putusan')
                    ax.set_xlabel('Periode (Tahun-Bulan)')
                    ax.set_ylabel('Jumlah Kasus Putusan')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                else:
                    st.warning("Data tanggal tidak tersedia atau tidak valid untuk analisis Tren.")

        # ====================================================================
        # TAB 2: FAKTOR & USIA (Insight Dominan)
        # ====================================================================
        with tab2:
            st.header("Analisis Faktor Dominan & Rentang Usia")
            
            # A. Faktor Dominan
            st.subheader("Top 5 Faktor Penyebab Perceraian Terbanyak")
            faktor_counts = df['faktor'].value_counts().head(5)
            
            col_f1, col_f2 = st.columns([1, 2])
            with col_f1:
                st.dataframe(faktor_counts)
            with col_f2:
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.barplot(x=faktor_counts.values, y=faktor_counts.index, palette='viridis', ax=ax)
                ax.set_title('Top 5 Faktor Penyebab Perceraian')
                ax.set_xlabel('Jumlah Kasus')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
            st.markdown("---")
            
            # B. Kategori Usia
            st.subheader("Distribusi Rentang Usia Pasangan")
            bins = [0, 20, 30, 40, 50, 60, 100]
            labels = ['<20', '21-30', '31-40', '41-50', '51-60', '>60']
            
            df['rentang_usia_p'] = pd.cut(df['umur_p'], bins=bins, labels=labels)
            df['rentang_usia_t'] = pd.cut(df['umur_t'], bins=bins, labels=labels)
            
            col_u1, col_u2 = st.columns(2)
            with col_u1:
                st.write("**Rentang Usia Penggugat (P):**")
                st.dataframe(df['rentang_usia_p'].value_counts())
            with col_u2:
                st.write("**Rentang Usia Tergugat (T):**")
                st.dataframe(df['rentang_usia_t'].value_counts())
                
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            sns.countplot(data=df, x='rentang_usia_p', ax=axes[0], palette='Blues_r', order=labels)
            axes[0].set_title('Distribusi Usia Penggugat')
            axes[0].set_xlabel('Rentang Usia')
            
            sns.countplot(data=df, x='rentang_usia_t', ax=axes[1], palette='Oranges_r', order=labels)
            axes[1].set_title('Distribusi Usia Tergugat')
            axes[1].set_xlabel('Rentang Usia')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # ====================================================================
        # TAB 3: CLUSTERING (K-Means)
        # ====================================================================
        with tab3:
            st.header("Clustering Pola Perceraian (K-Means)")
            st.markdown("Mengelompokkan data berdasarkan pola interaksi usia antara Penggugat dan Tergugat.")
            
            X = df[['umur_p', 'umur_t']].copy()
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Slider interaktif agar user bisa menentukan sendiri jumlah cluster
            n_clusters = st.slider("Pilih Jumlah Cluster (K):", min_value=2, max_value=5, value=3)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            df['cluster'] = kmeans.fit_predict(X_scaled)
            
            cluster_profile = df.groupby('cluster')[['umur_p', 'umur_t']].mean()
            cluster_size = df['cluster'].value_counts()
            
            col_c1, col_c2 = st.columns([1, 2])
            with col_c1:
                st.write("**Profil Rata-rata Umur per Cluster:**")
                st.dataframe(cluster_profile)
                st.write("**Jumlah Kasus per Cluster:**")
                st.dataframe(cluster_size)
                
            with col_c2:
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.scatterplot(data=df, x='umur_p', y='umur_t', hue='cluster', palette='Set1', alpha=0.6, ax=ax)
                ax.set_title('Clustering Pola Perceraian Berdasarkan Usia Pasangan')
                ax.set_xlabel('Usia Penggugat')
                ax.set_ylabel('Usia Tergugat')
                ax.legend(title='Cluster')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        # ====================================================================
        # TAB 4: DATA MENTAH
        # ====================================================================
        with tab4:
            st.header("Tampilan Data Mentah (Cleaned)")
            st.dataframe(df)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses data: {e}")
        st.info("Pastikan struktur dan pemisah kolom (delimiter ';') pada file CSV Anda sudah sesuai.")

else:
    st.info("💡 Silakan unggah file CSV `20260312_perkara_cerai.csv` di panel sebelah kiri untuk memulai analisis.")