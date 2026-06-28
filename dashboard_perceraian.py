"""
Dashboard Visualisasi Interaktif - Analisis Pola Perkara Perceraian
====================================================================
Dashboard ini merangkum semua temuan utama dalam satu tampilan terpadu:
1. Peta Distribusi Wilayah
2. Tren Temporal Kasus
3. Distribusi Alasan Perceraian
4. Profil Cluster Pasangan

Menjalankan: python dashboard_perceraian.py
Akses dashboard di: http://127.0.0.1:8050
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import re


# ============================================================================
# 1. DATA PREPROCESSING
# ============================================================================

def load_and_clean_data():
    df = pd.read_csv("20260312_perkara_cerai.csv", sep=';', encoding='latin-1')

    df['umur_p'] = pd.to_numeric(df['umur_p'], errors='coerce')
    df['umur_t'] = pd.to_numeric(df['umur_t'], errors='coerce')
    df['lama_pernikahan_bulan'] = pd.to_numeric(df['lama_pernikahan_bulan'], errors='coerce')
    df['tanggal_putusan'] = pd.to_datetime(df['tanggal_putusan'], errors='coerce')

    for col in ['kecamatan_p', 'kabupaten_p', 'kecamatan_t', 'kabupaten_t']:
        df[col] = df[col].astype(str).str.strip().str.rstrip('.').str.rstrip(',')
        df[col] = df[col].apply(clean_region)

    df = df.dropna(subset=['umur_p', 'umur_t'])
    return df


def clean_region(text):
    if pd.isna(text) or text in ['0', 'nan', '']:
        return pd.NA

    text = re.sub(r'DOMISILI.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Alamat.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'RT.*', '', text)
    text = re.sub(r'RW.*', '', text)
    text = re.sub(r'Blok.*', '', text)
    text = re.sub(r'Desa.*', '', text)
    text = re.sub(r'Perum.*', '', text)
    text = re.sub(r'Kec\.', 'Kecamatan', text)
    text = re.sub(r'[.!?,;]+$', '', text).strip()
    text = re.sub(r'\s+', ' ', text).strip()

    return text if text and len(text) > 2 else pd.NA


def create_age_groups(df):
    bins = [0, 20, 30, 40, 50, 60, 100]
    labels = ['<20', '21-30', '31-40', '41-50', '51-60', '>60']
    df['rentang_usia_p'] = pd.cut(df['umur_p'], bins=bins, labels=labels)
    df['rentang_usia_t'] = pd.cut(df['umur_t'], bins=bins, labels=labels)
    return df


def perform_clustering(df, n_clusters=3):
    X = df[['umur_p', 'umur_t']].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    cluster_labels = {}
    for i, center in enumerate(centers):
        avg_age = (center[0] + center[1]) / 2
        if avg_age < 35:
            cluster_labels[i] = "Pasangan Muda"
        elif avg_age < 45:
            cluster_labels[i] = "Pasangan Menengah"
        else:
            cluster_labels[i] = "Pasangan Dewasa"

    df['cluster_label'] = df['cluster'].map(cluster_labels)
    return df, cluster_labels


def create_marriage_duration_groups(df):
    bins = [0, 12, 36, 60, 120, 360, 9999]
    labels = ['<1 tahun', '1-3 tahun', '3-5 tahun', '5-10 tahun', '10-30 tahun', '>30 tahun']
    df['kategori_lama_nikah'] = pd.cut(df['lama_pernikahan_bulan'], bins=bins, labels=labels)
    return df


# ============================================================================
# 2. VISUALIZATION FUNCTIONS
# ============================================================================

def create_region_map(df):
    kabupaten_counts = df['kabupaten_p'].value_counts().reset_index()
    kabupaten_counts.columns = ['Kabupaten', 'Jumlah Kasus']
    top_15 = kabupaten_counts.head(15)

    fig = px.bar(
        top_15,
        x='Jumlah Kasus',
        y='Kabupaten',
        orientation='h',
        color='Jumlah Kasus',
        color_continuous_scale='YlOrRd',
        title='Top 15 Kabupaten/Kota - Distribusi Kasus Perceraian'
    )
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        height=500,
        coloraxis_showscale=False
    )
    return fig


def create_kecamatan_treemap(df):
    kecamatan_counts = df['kecamatan_p'].value_counts().reset_index()
    kecamatan_counts.columns = ['Kecamatan', 'Jumlah Kasus']
    top_30 = kecamatan_counts.head(30)

    fig = px.treemap(
        top_30,
        path=['Kecamatan'],
        values='Jumlah Kasus',
        color='Jumlah Kasus',
        color_continuous_scale='Blues',
        title='Treemap Kecamatan - Top 30 Daerah Kasus Terbanyak'
    )
    fig.update_layout(height=500)
    return fig


def create_temporal_trend(df):
    df_ts = df.dropna(subset=['tanggal_putusan']).copy()

    if df_ts.empty:
        fig = go.Figure()
        fig.add_annotation(text="Data tanggal tidak valid", x=0.5, y=0.5, showarrow=False)
        return fig

    df_ts['tahun_bulan'] = df_ts['tanggal_putusan'].dt.to_period('M')
    trend_bulanan = df_ts.groupby('tahun_bulan').size().reset_index()
    trend_bulanan.columns = ['Periode', 'Jumlah Kasus']
    trend_bulanan['Periode'] = trend_bulanan['Periode'].astype(str)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=trend_bulanan['Periode'],
        y=trend_bulanan['Jumlah Kasus'],
        mode='lines+markers',
        name='Tren Bulanan',
        line=dict(color='#e74c3c', width=2),
        marker=dict(size=6)
    ))

    if len(trend_bulanan) >= 3:
        moving_avg = trend_bulanan['Jumlah Kasus'].rolling(window=3).mean()
        fig.add_trace(go.Scatter(
            x=trend_bulanan['Periode'],
            y=moving_avg,
            mode='lines',
            name='Moving Average (3 bulan)',
            line=dict(color='#3498db', width=2, dash='dash')
        ))

    fig.update_layout(
        title='Tren Jumlah Kasus Perceraian per Bulan',
        xaxis_title='Periode',
        yaxis_title='Jumlah Kasus',
        height=500,
        hovermode='x unified'
    )
    return fig


def create_yearly_bar(df):
    df_ts = df.dropna(subset=['tanggal_putusan']).copy()

    if df_ts.empty:
        return go.Figure()

    df_ts['tahun'] = df_ts['tanggal_putusan'].dt.year
    yearly = df_ts.groupby('tahun').size().reset_index()
    yearly.columns = ['Tahun', 'Jumlah Kasus']

    fig = px.bar(
        yearly,
        x='Tahun',
        y='Jumlah Kasus',
        color='Jumlah Kasus',
        color_continuous_scale='Viridis',
        title='Jumlah Kasus Perceraian per Tahun',
        text='Jumlah Kasus'
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(height=400, coloraxis_showscale=False)
    return fig


def create_faktor_distribution(df):
    faktor_counts = df['faktor'].value_counts().reset_index()
    faktor_counts.columns = ['Faktor Penyebab', 'Jumlah Kasus']
    total = faktor_counts['Jumlah Kasus'].sum()
    faktor_counts['Persentase'] = (faktor_counts['Jumlah Kasus'] / total * 100).round(1)
    top_10 = faktor_counts.head(10)

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "pie"}, {"type": "bar"}]],
        subplot_titles=('Persentase Faktor', 'Jumlah Kasus per Faktor')
    )

    fig.add_trace(
        go.Pie(
            labels=top_10['Faktor Penyebab'],
            values=top_10['Jumlah Kasus'],
            hole=0.4,
            textinfo='percent',
            marker=dict(colors=px.colors.qualitative.Set3)
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(
            x=top_10['Jumlah Kasus'],
            y=top_10['Faktor Penyebab'],
            orientation='h',
            marker=dict(color=top_10['Jumlah Kasus'], colorscale='YlGnBu'),
            text=top_10['Persentase'].astype(str) + '%',
            textposition='outside'
        ),
        row=1, col=2
    )

    fig.update_layout(
        title_text='Distribusi Faktor/Alasan Perceraian',
        height=500,
        showlegend=False,
        yaxis2={'categoryorder': 'total ascending'}
    )
    return fig


def create_jenis_perkara_chart(df):
    jenis_counts = df['jenis_perkara_nama'].value_counts().reset_index()
    jenis_counts.columns = ['Jenis Perkara', 'Jumlah Kasus']

    fig = px.pie(
        jenis_counts,
        names='Jenis Perkara',
        values='Jumlah Kasus',
        title='Proporsi Jenis Perceraian',
        color_discrete_map={'Cerai Gugat': '#3498db', 'Cerai Talak': '#e74c3c'},
        hole=0.5
    )
    fig.update_traces(textinfo='percent+label+value')
    fig.update_layout(height=400)
    return fig


def create_age_distribution(df):
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Distribusi Usia Penggugat', 'Distribusi Usia Tergugat')
    )

    fig.add_trace(
        go.Histogram(
            x=df['umur_p'].dropna(),
            nbinsx=30,
            name='Penggugat',
            marker_color='#3498db',
            opacity=0.8
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Histogram(
            x=df['umur_t'].dropna(),
            nbinsx=30,
            name='Tergugat',
            marker_color='#e74c3c',
            opacity=0.8
        ),
        row=1, col=2
    )

    fig.update_layout(
        title_text='Distribusi Usia Pasangan',
        height=400,
        showlegend=True,
        xaxis_title='Usia',
        xaxis2_title='Usia',
        yaxis_title='Frekuensi',
        yaxis2_title='Frekuensi'
    )
    return fig


def create_age_group_comparison(df):
    usia_p = df['rentang_usia_p'].value_counts().reset_index()
    usia_p.columns = ['Rentang Usia', 'Jumlah']
    usia_p['Tipe'] = 'Penggugat'

    usia_t = df['rentang_usia_t'].value_counts().reset_index()
    usia_t.columns = ['Rentang Usia', 'Jumlah']
    usia_t['Tipe'] = 'Tergugat'

    combined = pd.concat([usia_p, usia_t])

    fig = px.bar(
        combined,
        x='Rentang Usia',
        y='Jumlah',
        color='Tipe',
        barmode='group',
        title='Perbandingan Rentang Usia Penggugat vs Tergugat',
        color_discrete_map={'Penggugat': '#3498db', 'Tergugat': '#e74c3c'}
    )
    fig.update_layout(height=400)
    return fig


def create_marriage_duration_chart(df):
    nikah_counts = df['kategori_lama_nikah'].value_counts().reset_index()
    nikah_counts.columns = ['Lama Pernikahan', 'Jumlah Kasus']

    order = ['<1 tahun', '1-3 tahun', '3-5 tahun', '5-10 tahun', '10-30 tahun', '>30 tahun']
    nikah_counts['sort_key'] = nikah_counts['Lama Pernikahan'].apply(
        lambda x: order.index(x) if x in order else 99
    )
    nikah_counts = nikah_counts.sort_values('sort_key')

    fig = px.bar(
        nikah_counts,
        x='Lama Pernikahan',
        y='Jumlah Kasus',
        color='Jumlah Kasus',
        color_continuous_scale='OrRd',
        title='Distribusi Lama Pernikahan Sebelum Cerai'
    )
    fig.update_layout(height=400, coloraxis_showscale=False)
    return fig


def create_cluster_scatter(df):
    colors = {'Pasangan Muda': '#2ecc71', 'Pasangan Menengah': '#f39c12', 'Pasangan Dewasa': '#e74c3c'}

    fig = go.Figure()

    for cluster_name in df['cluster_label'].unique():
        cluster_data = df[df['cluster_label'] == cluster_name]
        fig.add_trace(go.Scatter(
            x=cluster_data['umur_p'],
            y=cluster_data['umur_t'],
            mode='markers',
            name=cluster_name,
            marker=dict(
                color=colors.get(cluster_name, '#95a5a6'),
                size=6,
                opacity=0.6
            ),
            hovertemplate='<b>Penggugat:</b> %{x} tahun<br><b>Tergugat:</b> %{y} tahun'
        ))

    max_age = max(df['umur_p'].max(), df['umur_t'].max())
    fig.add_trace(go.Scatter(
        x=[0, max_age],
        y=[0, max_age],
        mode='lines',
        name='Garis Kesetaraan Usia',
        line=dict(color='gray', dash='dash')
    ))

    fig.update_layout(
        title='Clustering Pola Perceraian Berdasarkan Usia Pasangan',
        xaxis_title='Usia Penggugat (tahun)',
        yaxis_title='Usia Tergugat (tahun)',
        height=500,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig


def create_cluster_profile_cards(df):
    cards = []
    colors = {'Pasangan Muda': 'success', 'Pasangan Menengah': 'warning', 'Pasangan Dewasa': 'danger'}
    icons = {'Pasangan Muda': 'Umur Muda', 'Pasangan Menengah': 'Umur Menengah', 'Pasangan Dewasa': 'Umur Dewasa'}

    for cluster_name in sorted(df['cluster_label'].unique()):
        cluster_data = df[df['cluster_label'] == cluster_name]
        count = len(cluster_data)
        pct = count / len(df) * 100
        avg_p = cluster_data['umur_p'].mean()
        avg_t = cluster_data['umur_t'].mean()
        avg_nikah = cluster_data['lama_pernikahan_bulan'].mean()

        card = dbc.Card(
            dbc.CardBody([
                html.H5(f"{icons.get(cluster_name, 'Cluster')} - {cluster_name}", className="card-title"),
                html.H3(f"{count:,} kasus ({pct:.1f}%)", className="text-primary"),
                html.Hr(),
                html.P([
                    html.Strong("Rata-rata Usia:"),
                    html.Br(),
                    f"  Penggugat: {avg_p:.1f} tahun",
                    html.Br(),
                    f"  Tergugat: {avg_t:.1f} tahun",
                ]),
                html.P([
                    html.Strong("Rata-rata Lama Nikah:"),
                    f" {avg_nikah/12:.1f} tahun ({avg_nikah:.0f} bulan)"
                ]),
            ]),
            className="mb-3",
            outline=True,
            color=colors.get(cluster_name, 'secondary')
        )
        cards.append(card)

    return cards


def create_faktor_by_age_heatmap(df):
    top_faktor = df['faktor'].value_counts().head(5).index.tolist()
    df_top = df[df['faktor'].isin(top_faktor)].copy()

    cross = pd.crosstab(df_top['rentang_usia_p'], df_top['faktor'])

    fig = px.imshow(
        cross,
        labels=dict(x="Faktor Penyebab", y="Rentang Usia Penggugat", color="Jumlah Kasus"),
        color_continuous_scale='YlOrRd',
        title='Heatmap: Faktor Perceraian vs Rentang Usia Penggugat',
        text_auto=True
    )
    fig.update_layout(height=400)
    return fig


# ============================================================================
# 3. DASHBOARD LAYOUT & APP
# ============================================================================

def create_dashboard():
    print("Memuat dan memproses data...")
    df = load_and_clean_data()
    df = create_age_groups(df)
    df, cluster_labels = perform_clustering(df)
    df = create_marriage_duration_groups(df)

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.FLATLY],
        title="Dashboard Analisis Perceraian"
    )

    # Statistik ringkas
    total_kasus = len(df)
    avg_umur_p = df['umur_p'].mean()
    avg_umur_t = df['umur_t'].mean()
    avg_nikah = df['lama_pernikahan_bulan'].mean()
    pct_gugat = (df['jenis_perkara_nama'] == 'Cerai Gugat').mean() * 100
    top_alasan = df['faktor'].value_counts().index[0]

    stats_cards = dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4(f"{total_kasus:,}", className="text-primary"),
                html.P("Total Kasus")
            ])
        ], outline=True, color="primary"), width=2),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4(f"{avg_umur_p:.1f} thn", className="text-info"),
                html.P("Rata-rata Usia Penggugat")
            ])
        ], outline=True, color="info"), width=2),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4(f"{avg_umur_t:.1f} thn", className="text-warning"),
                html.P("Rata-rata Usia Tergugat")
            ])
        ], outline=True, color="warning"), width=2),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4(f"{avg_nikah/12:.1f} thn", className="text-success"),
                html.P("Rata-rata Lama Nikah")
            ])
        ], outline=True, color="success"), width=2),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4(f"{pct_gugat:.0f}%", className="text-danger"),
                html.P("Proporsi Cerai Gugat")
            ])
        ], outline=True, color="danger"), width=2),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6(top_alasan[:25] + "...", className="text-secondary"),
                html.P("Faktor Dominan")
            ])
        ], outline=True, color="secondary"), width=2),
    ], className="mb-4")

    cluster_cards = create_cluster_profile_cards(df)

    # Layout utama
    app.layout = dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H1("Dashboard Analisis Pola Perceraian", className="text-primary mt-4"),
                html.P("Visualisasi interaktif mengungkap pola dan tren kasus perceraian", className="lead"),
                html.Hr()
            ])
        ]),

        # Statistik Ringkas
        stats_cards,

        # Tab Navigation
        dbc.Tabs([
            # ==================== TAB 1: DISTRIBUSI WILAYAH ====================
            dbc.Tab(label=" Distribusi Wilayah", tab_id="tab-wilayah", children=[
                dbc.Row([
                    dbc.Col([
                        html.H3("Peta Distribusi Kasus per Kabupaten/Kota", className="mt-4"),
                        html.P("Visualisasi top 15 wilayah dengan jumlah kasus perceraian terbanyak"),
                        dcc.Graph(figure=create_region_map(df))
                    ], width=12)
                ]),
                dbc.Row([
                    dbc.Col([
                        html.H3("Treemap Kecamatan", className="mt-4"),
                        html.P("Hierarki kecamatan dengan kasus terbanyak"),
                        dcc.Graph(figure=create_kecamatan_treemap(df))
                    ], width=12)
                ]),
            ]),

            # ==================== TAB 2: TREN TEMPORAL ====================
            dbc.Tab(label=" Tren Temporal", tab_id="tab-temporal", children=[
                dbc.Row([
                    dbc.Col([
                        html.H3("Tren Kasus Perceraian per Bulan", className="mt-4"),
                        html.P("Garis tren menunjukkan pola kenaikan/penurunan kasus dari waktu ke waktu"),
                        dcc.Graph(figure=create_temporal_trend(df))
                    ], width=12)
                ]),
                dbc.Row([
                    dbc.Col([
                        html.H3("Distribusi per Tahun", className="mt-4"),
                        dcc.Graph(figure=create_yearly_bar(df))
                    ], width=12)
                ]),
            ]),

            # ==================== TAB 3: DISTRIBUSI ALASAN ====================
            dbc.Tab(label=" Distribusi Alasan", tab_id="tab-alasan", children=[
                dbc.Row([
                    dbc.Col([
                        html.H3("Faktor/Alasan Perceraian", className="mt-4"),
                        dcc.Graph(figure=create_faktor_distribution(df))
                    ], width=12)
                ]),
                dbc.Row([
                    dbc.Col([
                        html.H4("Jenis Perceraian"),
                        dcc.Graph(figure=create_jenis_perkara_chart(df))
                    ], width=6),
                    dbc.Col([
                        html.H4("Lama Pernikahan Sebelum Cerai"),
                        dcc.Graph(figure=create_marriage_duration_chart(df))
                    ], width=6),
                ]),
                dbc.Row([
                    dbc.Col([
                        html.H3("Heatmap Faktor vs Usia", className="mt-4"),
                        dcc.Graph(figure=create_faktor_by_age_heatmap(df))
                    ], width=12)
                ]),
            ]),

            # ==================== TAB 4: PROFIL CLUSTER ====================
            dbc.Tab(label=" Profil Cluster", tab_id="tab-cluster", children=[
                dbc.Row([
                    dbc.Col([
                        html.H3("Pola Cluster Pasangan", className="mt-4"),
                        html.P("Scatter plot menunjukkan pengelompokan pasangan berdasarkan usia"),
                        dcc.Graph(figure=create_cluster_scatter(df))
                    ], width=12)
                ]),
                dbc.Row([
                    dbc.Col([html.H3("Profil Tiap Cluster", className="mt-4")], width=12)
                ]),
                dbc.Row([
                    dbc.Col(cluster_cards[i] if i < len(cluster_cards) else [], width=4)
                    for i in range(3)
                ], className="mb-3"),
                dbc.Row([
                    dbc.Col([
                        html.H3("Distribusi Usia", className="mt-4"),
                        dcc.Graph(figure=create_age_distribution(df))
                    ], width=12)
                ]),
                dbc.Row([
                    dbc.Col([
                        html.H4("Perbandingan Rentang Usia"),
                        dcc.Graph(figure=create_age_group_comparison(df))
                    ], width=12)
                ]),
            ]),

            # ==================== TAB 5: DATA TABLE ====================
            dbc.Tab(label=" Tabel Data", tab_id="tab-data", children=[
                dbc.Row([
                    dbc.Col([
                        html.H3("Data Perceraian", className="mt-4"),
                        dash_table.DataTable(
                            columns=[
                                {"name": col, "id": col}
                                for col in ['perkara_id', 'tanggal_putusan', 'jenis_perkara_nama',
                                           'umur_p', 'umur_t', 'lama_pernikahan_bulan', 'faktor',
                                           'kecamatan_p', 'kabupaten_p', 'cluster_label']
                            ],
                            data=df.head(500).to_dict('records'),
                            page_size=20,
                            style_table={'overflowX': 'auto'},
                            style_cell={
                                'textAlign': 'left',
                                'padding': '10px',
                                'fontSize': '12px'
                            },
                            style_header={
                                'backgroundColor': '#2c3e50',
                                'color': 'white',
                                'fontWeight': 'bold'
                            },
                            style_data_conditional=[
                                {'if': {'row_index': 'odd'}, 'backgroundColor': '#f8f9fa'}
                            ],
                            filter_action="native",
                            sort_action="native",
                            page_action="native",
                        )
                    ], width=12)
                ]),
            ]),

        ], id="tabs", active_tab="tab-wilayah"),

        # Footer
        dbc.Row([
            dbc.Col([
                html.Hr(),
                html.P(
                    "Dashboard Analisis Pola Perceraian | Dibuat dengan Plotly Dash",
                    className="text-center text-muted"
                )
            ])
        ], className="mt-4 mb-4")
    ], fluid=True)

    return app


# ============================================================================
# 4. MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    app = create_dashboard()
    print("\n" + "="*60)
    print("DASHBOARD BERHASIL DIJALANKAN!")
    print("="*60)
    print("Akses dashboard di: http://127.0.0.1:8050")
    print("Tekan Ctrl+C untuk menghentikan server")
    print("="*60 + "\n")
    app.run(debug=True, host='127.0.0.1', port=8050)
