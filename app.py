import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# -----------------------------
# Config
# -----------------------------
st.set_page_config(
    page_title="Netflix Recommender (Content-Based)",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_DATA_PATH = Path(__file__).parent / "netflix_titles.csv"

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
    /* Main styling */
    .main-header {
        background: linear-gradient(135deg, #E50914 0%, #B81D24 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.1rem;
    }
    
    /* Card styling */
    .recommendation-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #E50914;
        transition: transform 0.3s ease;
    }
    
    .recommendation-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.15);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid #dee2e6;
    }
    
    .metric-card h3 {
        color: #E50914;
        font-size: 2rem !important;
        margin-bottom: 0.5rem;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2d2d2d 0%, #1a1a1a 100%);
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #E50914 0%, #B81D24 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(229, 9, 20, 0.3);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px 8px 0 0;
        padding: 1rem 2rem;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #E50914 !important;
        color: white !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #E50914 0%, #B81D24 100%);
    }
    
    /* Badge styling */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        background: #E50914;
        color: white;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    /* Similarity score */
    .similarity-score {
        background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        display: inline-block;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    /* Custom expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 8px !important;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Helpers
# -----------------------------
def _normalize_text(x: object) -> str:
    """Normalize text for vectorization."""
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"unknown", "nan", "none", "null"}:
        return ""
    s = s.replace("&", " and ")
    s = s.lower()
    s = re.sub(r"[^0-9a-z]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _safe_str(x: object) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    s = str(x)
    if s.strip().lower() in {"unknown", "nan", "none", "null"}:
        return ""
    return s

@st.cache_data(show_spinner=False)
def load_data_from_path(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str)

@st.cache_data(show_spinner=False)
def load_data_from_upload(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))

@st.cache_data(show_spinner=False)
def prepare_data(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    expected = [
        "show_id", "type", "title", "director", "cast", "country", 
        "date_added_iso", "release_year", "rating", "duration", 
        "listed_in", "description"
    ]
    for col in expected:
        if col not in df.columns:
            df[col] = ""
    
    text_cols = ["type", "title", "director", "cast", "country", "rating", "duration", "listed_in", "description"]
    for c in text_cols:
        df[c] = df[c].fillna("").astype(str)
        df[c] = df[c].replace({"Unknown": ""})
    
    if "release_year" in df.columns:
        df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").fillna(0).astype(int)
    else:
        df["release_year"] = 0
    
    df["soup"] = (
        df["title"].map(_normalize_text) + " " +
        df["type"].map(_normalize_text) + " " +
        df["director"].map(_normalize_text) + " " +
        df["cast"].map(_normalize_text) + " " +
        df["country"].map(_normalize_text) + " " +
        df["listed_in"].map(_normalize_text) + " " +
        df["rating"].map(_normalize_text) + " " +
        df["description"].map(_normalize_text)
    ).str.strip()
    
    df["display_title"] = df["title"].astype(str) + " (" + df["type"].astype(str) + ", " + df["release_year"].astype(str) + ")"
    dup = df["display_title"].duplicated(keep=False)
    if dup.any():
        df.loc[dup, "display_title"] = df.loc[dup].apply(
            lambda r: f"{r['title']} ({r['type']}, {r['release_year']}) — {r.get('show_id','')}",
            axis=1,
        )
    
    if df["show_id"].astype(str).duplicated().any():
        df["show_id"] = df.apply(lambda r: f"{r.get('show_id','')}_{r.name}", axis=1)
    
    return df

@st.cache_resource(show_spinner=False)
def build_vectorizer_and_matrix(corpus: pd.Series):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(corpus.values)
    return vectorizer, tfidf_matrix

def recommend_by_index(
    idx: int,
    df: pd.DataFrame,
    tfidf_matrix,
    top_n: int = 10,
    same_type: bool = True,
    year_min: int | None = None,
    year_max: int | None = None,
):
    sims = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
    order = sims.argsort()[::-1]
    order = order[order != idx]

    recs = df.iloc[order].copy()
    recs["similarity"] = sims[order]

    if same_type:
        selected_type = df.iloc[idx]["type"]
        recs = recs[recs["type"] == selected_type]

    if year_min is not None:
        recs = recs[recs["release_year"] >= year_min]
    if year_max is not None:
        recs = recs[recs["release_year"] <= year_max]

    return recs.head(top_n)

def recommend_by_query(
    query: str,
    df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    tfidf_matrix,
    top_n: int = 10,
    type_filter: str = "All",
    year_min: int | None = None,
    year_max: int | None = None,
):
    q = _normalize_text(query)
    if not q:
        return pd.DataFrame()

    q_vec = vectorizer.transform([q])
    if q_vec.nnz == 0:
        return pd.DataFrame()

    sims = linear_kernel(q_vec, tfidf_matrix).flatten()
    order = sims.argsort()[::-1]

    recs = df.iloc[order].copy()
    recs["similarity"] = sims[order]

    if type_filter != "All":
        recs = recs[recs["type"] == type_filter]
    if year_min is not None:
        recs = recs[recs["release_year"] >= year_min]
    if year_max is not None:
        recs = recs[recs["release_year"] <= year_max]

    return recs.head(top_n)

def split_and_count(series: pd.Series, sep: str = ",", top_k: int = 10) -> pd.Series:
    s = series.fillna("").astype(str).replace({"Unknown": ""})
    exploded = s.str.split(sep).explode().astype(str).str.strip()
    exploded = exploded[exploded != ""]
    return exploded.value_counts().head(top_k)

def create_recommendation_card(r: pd.Series, rank: int):
    """Create a beautiful recommendation card"""
    similarity = float(r.get("similarity", 0.0))
    progress_value = min(similarity * 1.5, 1.0)  # Scale for better visual
    
    card_html = f"""
    <div class="recommendation-card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <h4 style="margin: 0; color: #E50914;">{rank}. {_safe_str(r.get('title', ''))}</h4>
            <div class="similarity-score">
                {similarity:.1%}
            </div>
        </div>
        
        <div style="margin: 1rem 0;">
            <div style="display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 0.5rem;">
                <span class="badge">{_safe_str(r.get('type', ''))}</span>
                <span class="badge" style="background: #6c757d;">{r.get('release_year', '')}</span>
                <span class="badge" style="background: #17a2b8;">{_safe_str(r.get('rating', ''))}</span>
            </div>
            
            <div style="background: #f8f9fa; border-radius: 4px; padding: 0.5rem; margin: 0.5rem 0;">
                <strong>Genre:</strong> {_safe_str(r.get('listed_in', ''))}
            </div>
            
            <div style="font-size: 0.9rem; color: #666; margin: 0.5rem 0;">
                {_safe_str(r.get('description', ''))[:150]}...
            </div>
        </div>
    </div>
    """
    return card_html

def show_metric_card(title: str, value: str, subtitle: str = "", icon: str = "📊"):
    """Create a metric card"""
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
        <h3>{value}</h3>
        <div style="font-weight: 600; color: #495057;">{title}</div>
        <div style="font-size: 0.9rem; color: #6c757d;">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="main-header">
    <h1>🎬 Netflix Recommendation System</h1>
    <p>Content-Based Filtering menggunakan metadata film dan TV show untuk memberikan rekomendasi personal</p>
    <div style="display: flex; gap: 1rem; margin-top: 1rem;">
        <span class="badge" style="background: white; color: #E50914;">TF-IDF</span>
        <span class="badge" style="background: white; color: #E50914;">Cosine Similarity</span>
        <span class="badge" style="background: white; color: #E50914;">Content-Based</span>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <div style="font-size: 2rem; color: #E50914;">🎬</div>
        <h2 style="color: white; margin: 0;">Netflix Recommender</h2>
        <p style="color: rgba(255,255,255,0.7); font-size: 0.9rem;">Content-Based Filtering</p>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.radio(
        "Menu",
        ["🎯 Rekomendasi", "📊 Eksplorasi Dataset", "ℹ️ Tentang"],
        index=0,
        label_visibility="collapsed"
    )
    
    st.divider()
    
    st.markdown("#### ⚙️ Pengaturan Dataset")
    uploaded = st.file_uploader("Upload CSV File", type=["csv"], help="Upload dataset Netflix jika file lokal tidak tersedia")
    use_local = st.checkbox("Gunakan file lokal netflix_titles.csv", value=True)
    
    st.divider()
    
    st.markdown("#### 📈 Status Sistem")
    if 'raw_df' in locals():
        st.success("✅ Dataset siap digunakan")
    else:
        st.warning("⏳ Menunggu dataset...")
    
    st.caption("Made with ❤️ using Streamlit")

# Load data
raw_df = None
if uploaded is not None:
    raw_df = load_data_from_upload(uploaded.getvalue())
else:
    if use_local and DEFAULT_DATA_PATH.exists():
        raw_df = load_data_from_path(str(DEFAULT_DATA_PATH))
    else:
        st.warning(
            """
            ⚠️ Dataset belum tersedia.
            
            **Pilih salah satu opsi:**
            1. Upload file CSV di sidebar
            2. Taruh file **netflix_titles.csv** di folder yang sama dengan app.py
            3. Download dataset dari [Kaggle Netflix Dataset](https://www.kaggle.com/datasets/shivamb/netflix-shows)
            """
        )
        st.stop()

# Process data
with st.spinner("🔄 Memproses dataset..."):
    df = prepare_data(raw_df)
    vectorizer, tfidf_matrix = build_vectorizer_and_matrix(df["soup"])

# Common limits
min_year = int(df["release_year"].replace(0, np.nan).min(skipna=True) or 1900)
max_year = int(df["release_year"].max() or 2025)
type_options = ["Semua"] + sorted(df["type"].dropna().unique().tolist())

# -----------------------------
# Pages
# -----------------------------
if page == "🎯 Rekomendasi":
    tabs = st.tabs(["🎬 Berdasarkan Judul", "🔍 Berdasarkan Kata Kunci"])
    
    # Tab 1: By title
    with tabs[0]:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🎯 Pilih Konten")
            
            filter_type_for_selector = st.selectbox(
                "Filter berdasarkan tipe",
                options=type_options,
                index=0,
                help="Filter judul berdasarkan tipe (Movie/TV Show)"
            )
            
            if filter_type_for_selector == "Semua":
                selector_df = df
            else:
                selector_df = df[df["type"] == filter_type_for_selector]
            
            options = selector_df["display_title"].tolist()
            selected_display = st.selectbox(
                "Pilih judul untuk direkomendasikan",
                options=options,
                index=0,
                help="Pilih satu judul untuk mendapatkan rekomendasi serupa"
            )
            
            # Filters
            st.subheader("⚙️ Filter Rekomendasi")
            
            col_a, col_b = st.columns(2)
            with col_a:
                top_n = st.slider(
                    "Jumlah rekomendasi",
                    min_value=5,
                    max_value=20,
                    value=10,
                    step=1
                )
            
            with col_b:
                same_type = st.checkbox(
                    "Hanya tipe yang sama",
                    value=True,
                    help="Rekomendasikan hanya Movie↔Movie atau TV↔TV"
                )
            
            year_range = st.slider(
                "Rentang tahun rilis",
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year)
            )
            year_min, year_max = year_range
        
        with col2:
            st.subheader("📊 Statistik Dataset")
            
            show_metric_card(
                "Total Konten",
                f"{len(df):,}",
                "Movies & TV Shows"
            )
            
            show_metric_card(
                "Movies",
                f"{int((df['type']=='Movie').sum()):,}",
                "Film layar lebar"
            )
            
            show_metric_card(
                "TV Shows",
                f"{int((df['type']=='TV Show').sum()):,}",
                "Serial televisi"
            )
            
            # Recently added preview
            if "date_added_iso" in df.columns:
                st.subheader("🆕 Baru Ditambahkan")
                recent = df.copy()
                recent["date_added_iso"] = pd.to_datetime(recent["date_added_iso"], errors="coerce")
                recent = recent.sort_values("date_added_iso", ascending=False).head(3)
                
                for _, item in recent.iterrows():
                    with st.expander(f"{item['title']} ({item['release_year']})", expanded=False):
                        st.write(f"**Tipe:** {item['type']}")
                        st.write(f"**Genre:** {item['listed_in']}")
        
        # Recommendation button
        if st.button("🎯 Dapatkan Rekomendasi", type="primary", use_container_width=True):
            with st.spinner("🔍 Mencari rekomendasi..."):
                idx = int(df.index[df["display_title"] == selected_display][0])
                
                # Show selected item
                st.divider()
                st.subheader("🎬 Konten yang Dipilih")
                
                selected_item = df.iloc[idx]
                col_info1, col_info2 = st.columns([2, 1])
                
                with col_info1:
                    st.markdown(f"### {selected_item['title']}")
                    st.markdown(f"**{selected_item['type']} • {selected_item['release_year']} • {selected_item['rating']}**")
                    st.write(f"**Genre:** {selected_item['listed_in']}")
                    st.write(f"**Negara:** {selected_item['country']}")
                    
                with col_info2:
                    if selected_item['director']:
                        st.write(f"**Director:** {selected_item['director']}")
                    if selected_item['duration']:
                        st.write(f"**Durasi:** {selected_item['duration']}")
                
                if selected_item['description']:
                    st.write(f"**Sinopsis:** {selected_item['description']}")
                
                # Get recommendations
                recs = recommend_by_index(
                    idx=idx,
                    df=df,
                    tfidf_matrix=tfidf_matrix,
                    top_n=top_n,
                    same_type=same_type,
                    year_min=year_min,
                    year_max=year_max,
                )
                
                st.divider()
                st.subheader(f"🎯 {len(recs)} Rekomendasi Serupa")
                
                if recs.empty:
                    st.warning("Tidak ditemukan rekomendasi yang sesuai dengan filter.")
                else:
                    # Display as cards
                    for i, (_, r) in enumerate(recs.iterrows(), 1):
                        card_html = create_recommendation_card(r, i)
                        st.markdown(card_html, unsafe_allow_html=True)
                        
                        # Expanded details
                        with st.expander("📖 Lihat detail lengkap", expanded=False):
                            col_det1, col_det2 = st.columns(2)
                            with col_det1:
                                st.write(f"**Director:** {r.get('director', 'Tidak tersedia')}")
                                st.write(f"**Cast:** {r.get('cast', 'Tidak tersedia')}")
                                st.write(f"**Negara:** {r.get('country', 'Tidak tersedia')}")
                            with col_det2:
                                if r.get('description'):
                                    st.write("**Sinopsis:**")
                                    st.write(r['description'])
    
    # Tab 2: By keywords
    with tabs[1]:
        st.subheader("🔍 Cari dengan Kata Kunci")
        
        query = st.text_input(
            "Masukkan kata kunci pencarian:",
            placeholder="Contoh: romance comedy, sci-fi adventure, crime drama",
            help="Masukkan genre, tema, atau kata kunci yang ingin dicari"
        )
        
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            type_filter = st.selectbox(
                "Filter tipe",
                options=type_options,
                index=0
            )
        with col_f2:
            top_n_q = st.slider(
                "Jumlah hasil",
                min_value=5,
                max_value=20,
                value=10
            )
        with col_f3:
            st.write("")  # Spacing
            search_btn = st.button("🔍 Cari Konten", type="primary", use_container_width=True)
        
        year_range_q = st.slider(
            "Rentang tahun rilis",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year),
            key="year_q"
        )
        year_min_q, year_max_q = year_range_q
        
        if search_btn and query:
            with st.spinner("🔎 Mencari konten..."):
                recs_q = recommend_by_query(
                    query=query,
                    df=df,
                    vectorizer=vectorizer,
                    tfidf_matrix=tfidf_matrix,
                    top_n=top_n_q,
                    type_filter=type_filter,
                    year_min=year_min_q,
                    year_max=year_max_q,
                )
                
                if recs_q.empty:
                    st.error("""
                    ❌ Tidak ditemukan hasil untuk kata kunci tersebut.
                    
                    **Saran:**
                    - Gunakan kata kunci dalam bahasa Inggris
                    - Coba kata kunci yang lebih umum
                    - Kurangi filter yang diterapkan
                    """)
                else:
                    st.success(f"✅ Ditemukan {len(recs_q)} hasil untuk '{query}'")
                    
                    # Display results
                    for i, (_, r) in enumerate(recs_q.iterrows(), 1):
                        card_html = create_recommendation_card(r, i)
                        st.markdown(card_html, unsafe_allow_html=True)

elif page == "📊 Eksplorasi Dataset":
    st.header("📊 Eksplorasi Dataset Netflix")
    
    # Metrics row
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        show_metric_card("Total Konten", f"{len(df):,}", "Movies & TV Shows", "🎬")
    with col_m2:
        show_metric_card("Movies", f"{int((df['type']=='Movie').sum()):,}", "", "🎥")
    with col_m3:
        show_metric_card("TV Shows", f"{int((df['type']=='TV Show').sum()):,}", "", "📺")
    with col_m4:
        avg_year = int(df['release_year'].replace(0, np.nan).mean())
        show_metric_card("Tahun Rata-rata", str(avg_year), "", "📅")
    
    st.divider()
    
    # Sample data
    st.subheader("📋 Contoh Data")
    sample_size = st.slider("Jumlah baris contoh", 5, 50, 10)
    st.dataframe(
        df[["title", "type", "release_year", "rating", "duration", "listed_in"]].head(sample_size),
        use_container_width=True,
        hide_index=True
    )
    
    st.divider()
    
    # Charts
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        st.subheader("📈 Distribusi Tipe")
        type_counts = df["type"].value_counts()
        st.bar_chart(type_counts, color="#E50914")
        
        st.subheader("🌍 Top Negara")
        top_countries = split_and_count(df["country"], sep=",", top_k=10)
        st.bar_chart(top_countries, color="#17a2b8")
    
    with col_c2:
        st.subheader("🎭 Top Genre")
        top_genres = split_and_count(df["listed_in"], sep=",", top_k=10)
        st.bar_chart(top_genres, color="#28a745")
        
        st.subheader("📅 Tren Tahun Rilis")
        year_counts = df["release_year"].replace(0, np.nan).dropna().astype(int).value_counts().sort_index()
        st.line_chart(year_counts, color="#ffc107")
    
    st.divider()
    
    # Rating distribution
    st.subheader("⭐ Distribusi Rating")
    rating_counts = df["rating"].value_counts().head(15)
    st.bar_chart(rating_counts, color="#6f42c1")

elif page == "ℹ️ Tentang":
    col_about1, col_about2 = st.columns([2, 1])
    
    with col_about1:
        st.header("🎬 Tentang Sistem Rekomendasi Netflix")
        
        st.markdown("""
        ### 🎯 **Tujuan Proyek**
        Membangun sistem rekomendasi konten Netflix berbasis **konten** untuk membantu pengguna menemukan film dan TV show yang sesuai dengan preferensi mereka.
        
        ### 🛠️ **Teknologi yang Digunakan**
        
        **Algoritma:** Content-Based Filtering
        - **TF-IDF Vectorizer**: Mengubah teks menjadi representasi numerik
        - **Cosine Similarity**: Mengukur kemiripan antara konten
        - **Metadata Analysis**: Memanfaatkan genre, cast, director, deskripsi
        
        **Framework:**
        - Streamlit untuk UI/UX
        - Scikit-learn untuk machine learning
        - Pandas untuk data processing
        
        ### 📊 **Arsitektur Sistem**
        1. **Preprocessing Data**
           - Pembersihan dan normalisasi teks
           - Pembuatan 'soup' dari metadata
           
        2. **Feature Engineering**
           - TF-IDF Vectorization
           - Similarity matrix calculation
           
        3. **Recommendation Engine**
           - Content-based similarity matching
           - Filtering berdasarkan preferensi user
           
        ### 🔍 **Cara Kerja**
        1. Pilih judul atau masukkan kata kunci
        2. Sistem mencari konten dengan metadata serupa
        3. Hasil direkomendasikan berdasarkan tingkat kemiripan
        4. Filter dapat disesuaikan (tahun, tipe, dll)
        """)
    
    with col_about2:
        st.header("📈 Performa Sistem")
        
        show_metric_card(
            "Ukuran Dataset",
            f"{len(df):,}",
            "item konten"
        )
        
        show_metric_card(
            "Dimensi TF-IDF",
            f"{tfidf_matrix.shape[1]:,}",
            "fitur unik"
        )
        
        show_metric_card(
            "Waktu Response",
            "< 1 detik",
            "rata-rata"
        )
        
        st.divider()
        
        st.markdown("""
        ### 👨‍💻 **Developer**
        Sistem dikembangkan menggunakan:
        - Python 3.10+
        - Streamlit Cloud
        - Open-source libraries
        
        ### 📚 **Dataset Source**
        Netflix Movies and TV Shows dari Kaggle
        """)
    
    st.divider()
    
    # Feature highlights
    st.subheader("✨ Fitur Utama")
    
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; background: #f8f9fa; border-radius: 10px;">
            <div style="font-size: 2rem;">🎯</div>
            <h4>Rekomendasi Presisi</h4>
            <p>Berdasarkan kemiripan konten sebenarnya</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_f2:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; background: #f8f9fa; border-radius: 10px;">
            <div style="font-size: 2rem;">⚡</div>
            <h4>Responsif</h4>
            <p>Hasil instan dengan waktu loading minimal</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_f3:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; background: #f8f9fa; border-radius: 10px;">
            <div style="font-size: 2rem;">🎨</div>
            <h4>UI Modern</h4>
            <p>Antarmuka yang intuitif dan menarik</p>
        </div>
        """, unsafe_allow_html=True)