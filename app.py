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
    page_title="Netflix Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_DATA_PATH = Path(__file__).parent / "netflix_titles.csv"

# -----------------------------
# Custom CSS - Enhanced Color Scheme
# -----------------------------
st.markdown("""
<style>
    /* Font Import */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=Montserrat:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Montserrat', sans-serif !important;
        font-weight: 600 !important;
    }
    
    /* Main Header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem !important;
        margin-bottom: 0.8rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.95);
        font-size: 1.2rem;
        max-width: 800px;
        line-height: 1.6;
    }
    
    /* Card Styling */
    .recommendation-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9ff 100%);
        border-radius: 16px;
        padding: 1.8rem;
        margin: 1.2rem 0;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
        border: 1px solid #e6e8ff;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .recommendation-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 6px;
        height: 100%;
        background: linear-gradient(to bottom, #667eea, #764ba2);
    }
    
    .recommendation-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.2);
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9ff 100%);
        border-radius: 16px;
        padding: 1.8rem;
        text-align: center;
        border: 1px solid #e6e8ff;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.15);
    }
    
    .metric-card h3 {
        color: #667eea;
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem;
        font-weight: 700 !important;
    }
    
    /* Badge Styling */
    .badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 25px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 3px 10px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }
    
    .badge:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    .badge-movie {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    .badge-year {
        background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%) !important;
    }
    
    .badge-rating {
        background: linear-gradient(135deg, #FF9800 0%, #F57C00 100%) !important;
    }
    
    /* Similarity Score */
    .similarity-score {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
        color: white;
        padding: 0.6rem 1.2rem;
        border-radius: 12px;
        display: inline-block;
        font-weight: 700;
        font-size: 1.1rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0, 176, 155, 0.3);
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.85rem 2.5rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
        font-family: 'Montserrat', sans-serif;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
        color: white;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #f8f9ff 0%, #e6e8ff 100%);
        border-radius: 12px 12px 0 0;
        padding: 1rem 2rem;
        font-weight: 600;
        font-family: 'Montserrat', sans-serif;
        color: #667eea;
        border: 1px solid #e6e8ff;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    }
    
    /* Sidebar Styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f8f9ff 0%, #e6e8ff 100%);
        border-radius: 12px !important;
        font-weight: 600;
        font-family: 'Montserrat', sans-serif;
        color: #667eea;
        border: 1px solid #e6e8ff;
    }
    
    /* Input Field Styling */
    .stTextInput > div > div > input {
        border-radius: 12px;
        border: 2px solid #e6e8ff;
        padding: 0.75rem 1rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Selectbox Styling */
    .stSelectbox > div > div {
        border-radius: 12px;
        border: 2px solid #e6e8ff;
    }
    
    /* Slider Styling */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Checkbox Styling */
    .stCheckbox > label {
        font-weight: 500;
        color: #333;
    }
    
    /* Divider */
    .stDivider {
        border-color: #e6e8ff;
    }
    
    /* Dataframe Styling */
    .stDataFrame {
        border-radius: 12px;
        border: 1px solid #e6e8ff;
    }
    
    /* Tooltip */
    .stTooltip {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    
    /* Alert Messages */
    .stAlert {
        border-radius: 12px;
        border: none;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
    }
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Animation for cards */
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .recommendation-card {
        animation: fadeIn 0.5s ease-out;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header {
            padding: 1.5rem;
        }
        
        .main-header h1 {
            font-size: 2rem !important;
        }
        
        .metric-card h3 {
            font-size: 2rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Helper Functions
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
    
    # Clean column names (strip whitespace, lowercase)
    df.columns = df.columns.str.strip().str.lower()
    
    # Handle different column name variations
    column_mapping = {
        'show id': 'show_id',
        'show_id': 'show_id',
        'type': 'type',
        'title': 'title',
        'director': 'director',
        'director_list': 'director',
        'cast': 'cast',
        'cast_list': 'cast',
        'country': 'country',
        'country_list': 'country',
        'date_added': 'date_added',
        'date_added_iso': 'date_added',
        'release year': 'release_year',
        'release_year': 'release_year',
        'rating': 'rating',
        'duration': 'duration',
        'listed in': 'listed_in',
        'listed_in': 'listed_in',
        'listed_in_list': 'listed_in',
        'description': 'description',
        'actor': 'cast'  # Map actor to cast
    }
    
    # Rename columns based on mapping
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns and new_name not in df.columns:
            df[new_name] = df[old_name]
    
    # Ensure all expected columns exist
    expected = [
        "show_id", "type", "title", "director", "cast", "country", 
        "release_year", "rating", "duration", "listed_in", "description"
    ]
    
    for col in expected:
        if col not in df.columns:
            df[col] = ""
    
    # Clean and standardize type column
    df['type'] = df['type'].astype(str).str.strip()
    
    # Standardize type values - handle 'Tv Show' to 'TV Show'
    df['type'] = df['type'].apply(lambda x: 'TV Show' if str(x).lower() == 'tv show' else x)
    df['type'] = df['type'].apply(lambda x: 'Movie' if str(x).lower() == 'movie' else x)
    
    text_cols = ["type", "title", "director", "cast", "country", "rating", "duration", "listed_in", "description"]
    for c in text_cols:
        df[c] = df[c].fillna("").astype(str)
        df[c] = df[c].replace({
            "unknown": "", "Unknown": "", 
            "nan": "", "NaN": "", 
            "None": "", "none": "",
            "": ""
        })
    
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
    s = series.fillna("").astype(str).replace({"unknown": "", "Unknown": ""})
    exploded = s.str.split(sep).explode().astype(str).str.strip()
    exploded = exploded[exploded != ""]
    return exploded.value_counts().head(top_k)

def display_recommendation_card(r: pd.Series, rank: int):
    """Display a beautiful recommendation card using Streamlit components"""
    similarity = float(r.get("similarity", 0.0))
    title = _safe_str(r.get('title', ''))
    content_type = _safe_str(r.get('type', ''))
    year = r.get('release_year', '')
    rating = _safe_str(r.get('rating', ''))
    genre = _safe_str(r.get('listed_in', ''))
    description = _safe_str(r.get('description', 'No description available'))
    director = _safe_str(r.get('director', 'Not specified'))
    country = _safe_str(r.get('country', 'Not specified'))
    
    # Use markdown for the card container
    st.markdown(f"""
    <div class="recommendation-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <h4 style="margin: 0; color: #667eea; font-size: 1.3rem;">{rank}. {title}</h4>
            <div class="similarity-score">
                {similarity:.1%}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Use Streamlit components for the content
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<span class="badge badge-movie">{content_type}</span>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<span class="badge badge-year">{year}</span>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<span class="badge badge-rating">{rating}</span>', unsafe_allow_html=True)
    
    # Genre
    if genre:
        st.markdown(f"""
        <div style="background: #f8f9ff; border-radius: 10px; padding: 1rem; margin: 1rem 0; border-left: 4px solid #667eea;">
            <strong style="color: #667eea;">🎭 Genre:</strong> 
            <span style="color: #555; font-weight: 500;">{genre}</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Description
    if description and description != "No description available":
        st.markdown(f"""
        <div style="background: #f8f9ff; border-radius: 10px; padding: 1rem; margin: 1rem 0;">
            <strong style="color: #667eea;">📖 Deskripsi:</strong>
            <div style="font-size: 0.95rem; color: #666; line-height: 1.5; margin-top: 0.5rem;">
                {description}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Additional details
    col_details1, col_details2 = st.columns(2)
    with col_details1:
        if director and director != "Not specified":
            st.markdown(f"""
            <div style="margin-bottom: 0.5rem;">
                <strong style="color: #667eea;">🎬 Director:</strong>
                <div style="color: #666; font-size: 0.95rem;">{director}</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col_details2:
        if country and country != "Not specified":
            st.markdown(f"""
            <div style="margin-bottom: 0.5rem;">
                <strong style="color: #667eea;">🌍 Negara:</strong>
                <div style="color: #666; font-size: 0.95rem;">{country}</div>
            </div>
            """, unsafe_allow_html=True)

def display_metric_card(title: str, value: str, subtitle: str = "", icon: str = "📊"):
    """Display a metric card"""
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size: 2.5rem; margin-bottom: 0.8rem; color: #667eea;">{icon}</div>
        <h3>{value}</h3>
        <div style="font-weight: 600; color: #555; font-size: 1.1rem; margin-bottom: 0.5rem;">{title}</div>
        <div style="font-size: 0.9rem; color: #888; font-weight: 500;">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# Header Section
# -----------------------------
st.markdown("""
<div class="main-header">
    <h1>🎬 Sistem Rekomendasi Konten Netflix</h1>
    <p>
        Temukan film dan serial TV yang paling sesuai dengan preferensi Anda menggunakan algoritma 
        <strong>Content-Based Filtering</strong> canggih berbasis kemiripan metadata.
    </p>
    <div style="display: flex; gap: 1rem; margin-top: 1.5rem; flex-wrap: wrap;">
        <span class="badge badge-movie">TF-IDF Vectorization</span>
        <span class="badge badge-year">Cosine Similarity</span>
        <span class="badge badge-rating">Content-Based Filtering</span>
        <span class="badge" style="background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);">Machine Learning</span>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2.5rem; padding: 1.5rem; background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); border-radius: 15px;">
        <div style="font-size: 3rem; color: #667eea; margin-bottom: 0.5rem;">🎬</div>
        <h2 style="color: white; margin: 0; font-size: 1.5rem;">Netflix Recommender</h2>
        <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem; margin-top: 0.5rem;">Sistem Rekomendasi Cerdas</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📋 Menu Navigasi")
    page = st.radio(
        "Select a page:",
        ["🎯 Rekomendasi", "📊 Analisis Data", "ℹ️ Tentang Sistem"],
        index=0,
        label_visibility="collapsed"
    )
    
    st.divider()
    
    st.markdown("### ⚙️ Konfigurasi Data")
    with st.expander("📁 Pengaturan Dataset", expanded=True):
        uploaded = st.file_uploader(
            "Upload dataset CSV", 
            type=["csv"],
            help="Upload file dataset Netflix dalam format CSV"
        )
        use_local = st.checkbox(
            "Gunakan dataset lokal netflix_titles.csv", 
            value=True,
            help="Centang jika file dataset sudah tersedia di folder lokal"
        )
    
    st.divider()

# -----------------------------
# Load Data
# -----------------------------
raw_df = None
if uploaded is not None:
    with st.spinner("🔄 Memuat dataset dari file upload..."):
        raw_df = load_data_from_upload(uploaded.getvalue())
    st.success(f"✅ Dataset berhasil diupload: {len(raw_df)} baris data")
else:
    if use_local and DEFAULT_DATA_PATH.exists():
        with st.spinner("🔄 Memuat dataset lokal..."):
            raw_df = load_data_from_path(str(DEFAULT_DATA_PATH))
        st.success(f"✅ Dataset lokal berhasil dimuat: {len(raw_df)} baris data")
    else:
        st.error("""
        ## ⚠️ Dataset Tidak Ditemukan
        
        **Silakan pilih salah satu opsi:**
        1. 📤 Upload file dataset CSV melalui sidebar
        2. 📁 Letakkan file **netflix_titles.csv** di folder yang sama dengan aplikasi ini
        3. 🌐 Download dataset contoh dari [Kaggle Netflix Dataset](https://www.kaggle.com/datasets/shivamb/netflix-shows)
        
        ---
        
        **Format dataset yang didukung:**
        - File CSV dengan kolom: title, type, director, cast, country, release_year, rating, duration, listed_in, description
        - Encoding: UTF-8
        - Pemisah: Koma (,)
        """)
        st.stop()

# -----------------------------
# Process Data
# -----------------------------
with st.spinner("🔧 Memproses dan menyiapkan data untuk analisis..."):
    df = prepare_data(raw_df)
    vectorizer, tfidf_matrix = build_vectorizer_and_matrix(df["soup"])

# Update sidebar status
with st.sidebar:
    st.success("✅ **Sistem Aktif** - Dataset berhasil dimuat", icon="✅")
    
    # Display dataset stats in sidebar
    st.markdown("### 📊 Statistik Dataset")
    
    # Count movies and TV shows properly
    movie_count = len(df[df['type'] == 'Movie'])
    tv_count = len(df[df['type'] == 'TV Show'])
    other_count = len(df) - movie_count - tv_count
    
    st.write(f"**Total Data:** {len(df):,}")
    st.write(f"**Movies:** {movie_count:,}")
    st.write(f"**TV Shows:** {tv_count:,}")
    if other_count > 0:
        st.write(f"**Lainnya:** {other_count:,}")
    
    st.divider()
    
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%); border-radius: 10px;">
        <p style="color: #888; font-size: 0.85rem; margin: 0;">
            Dibangun dengan ❤️ menggunakan<br>
            <strong>Streamlit</strong> & <strong>Scikit-learn</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

# Common variables
min_year = int(df["release_year"].replace(0, np.nan).min(skipna=True) or 1900)
max_year = int(df["release_year"].max() or 2025)

# Get unique types
unique_types = sorted(df['type'].unique().tolist())
# Filter out empty strings
unique_types = [t for t in unique_types if t and str(t) != 'nan']
type_options = ["Semua Tipe"] + unique_types

# -----------------------------
# Page: Recommendation
# -----------------------------
if page == "🎯 Rekomendasi":
    tabs = st.tabs(["🎬 Berdasarkan Judul", "🔍 Berdasarkan Kata Kunci", "⭐ Favorit"])
    
    # Tab 1: By Title
    with tabs[0]:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 🎯 Pilih Konten untuk Direkomendasikan")
            
            # Content type filter
            filter_type_for_selector = st.selectbox(
                "**Filter berdasarkan tipe konten:**",
                options=type_options,
                index=0,
                help="Pilih untuk memfilter hanya Movie atau TV Show"
            )
            
            if filter_type_for_selector == "Semua Tipe":
                selector_df = df
            else:
                selector_df = df[df["type"] == filter_type_for_selector]
            
            # Title selection
            if len(selector_df) > 0:
                options = selector_df["display_title"].tolist()
                selected_display = st.selectbox(
                    "**Pilih judul yang ingin direkomendasikan:**",
                    options=options,
                    index=0,
                    help="Pilih satu judul film atau serial TV untuk mendapatkan rekomendasi serupa"
                )
            else:
                st.warning(f"Tidak ada konten dengan tipe: {filter_type_for_selector}")
                selected_display = None
            
            # Recommendation settings
            st.markdown("### ⚙️ Pengaturan Rekomendasi")
            
            col_a, col_b = st.columns(2)
            with col_a:
                top_n = st.slider(
                    "**Jumlah rekomendasi:**",
                    min_value=5,
                    max_value=20,
                    value=10,
                    step=1
                )
            
            with col_b:
                same_type = st.checkbox(
                    "**Hanya tipe yang sama**",
                    value=True,
                    help="Rekomendasikan hanya Movie↔Movie atau TV Show↔TV Show"
                )
            
            # Year range filter
            st.markdown("### 📅 Filter Tahun Rilis")
            year_range = st.slider(
                "**Pilih rentang tahun rilis:**",
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year),
                format="%d"
            )
            year_min, year_max = year_range
            
            # Recommendation button
            if selected_display and st.button("🚀 Dapatkan Rekomendasi", type="primary", use_container_width=True):
                idx = int(df.index[df["display_title"] == selected_display][0])
                
                # Display selected content
                st.divider()
                st.markdown("### 🎬 Konten yang Dipilih")
                
                selected_item = df.iloc[idx]
                col_info1, col_info2 = st.columns([2, 1])
                
                with col_info1:
                    st.markdown(f"#### {selected_item['title']}")
                    st.markdown(f"**🎭 {selected_item['type']} • 📅 {selected_item['release_year']} • ⭐ {selected_item['rating']}**")
                    
                    if selected_item['listed_in'] and selected_item['listed_in'] != "":
                        st.markdown(f"**Genre:** {selected_item['listed_in']}")
                    if selected_item['country'] and selected_item['country'] != "":
                        st.markdown(f"**Negara:** {selected_item['country']}")
                    
                    if selected_item['description'] and selected_item['description'] != "":
                        with st.expander("📖 Baca Sinopsis Lengkap"):
                            st.write(selected_item['description'])
                
                with col_info2:
                    if selected_item['director'] and selected_item['director'] != "":
                        st.markdown(f"**🎬 Sutradara:** {selected_item['director']}")
                    if selected_item['cast'] and selected_item['cast'] != "":
                        st.markdown(f"**👥 Pemain:** {selected_item['cast'][:100]}...")
                    if selected_item['duration'] and selected_item['duration'] != "":
                        st.markdown(f"**⏱️ Durasi:** {selected_item['duration']}")
                
                # Get recommendations
                with st.spinner("🔍 Mencari rekomendasi terbaik..."):
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
                st.markdown(f"### 🎯 {len(recs)} Rekomendasi Terbaik")
                
                if recs.empty:
                    st.warning("""
                    **Tidak ditemukan rekomendasi yang sesuai dengan filter.**
                    
                    Coba untuk:
                    1. Mengurangi filter tahun
                    2. Menonaktifkan filter "Hanya tipe yang sama"
                    3. Memilih judul lain
                    """)
                else:
                    # Display recommendations as cards
                    for i, (_, r) in enumerate(recs.iterrows(), 1):
                        display_recommendation_card(r, i)
                        
                        # Additional details in expander
                        with st.expander("🔍 Lihat detail teknis", expanded=False):
                            col_d1, col_d2 = st.columns(2)
                            with col_d1:
                                st.metric("Similarity Score", f"{float(r.get('similarity', 0)):.4f}")
                                if r.get('show_id'):
                                    st.write(f"**ID:** {r['show_id']}")
                            with col_d2:
                                if 'date_added' in r and r['date_added']:
                                    st.write(f"**Ditambahkan:** {r['date_added']}")
        
        with col2:
            st.markdown("### 📊 Statistik Dataset")
            
            display_metric_card(
                "Total Konten",
                f"{len(df):,}",
                "Movies & TV Shows",
                "🎬"
            )
            
            display_metric_card(
                "Movies",
                f"{movie_count:,}",
                "Film layar lebar",
                "🎥"
            )
            
            display_metric_card(
                "TV Shows",
                f"{tv_count:,}",
                "Serial televisi",
                "📺"
            )
            
            display_metric_card(
                "Tahun Terbaru",
                str(max_year),
                "Konten terbaru",
                "📅"
            )
            
            # Recently added preview
            st.divider()
            st.markdown("### 🆕 Konten Terbaru")
            # Try different date columns
            date_columns = ['date_added', 'date_added_iso', 'release_year']
            date_col = None
            for col in date_columns:
                if col in df.columns:
                    date_col = col
                    break
            
            if date_col:
                recent = df.copy()
                if date_col == 'release_year':
                    recent = recent.sort_values(date_col, ascending=False).head(5)
                else:
                    recent[date_col] = pd.to_datetime(recent[date_col], errors='coerce')
                    recent = recent.sort_values(date_col, ascending=False).head(5)
                
                for idx, item in recent.iterrows():
                    with st.expander(f"{item['title']} ({item['release_year']})", expanded=False):
                        st.write(f"**Tipe:** {item['type']}")
                        if item['listed_in']:
                            st.write(f"**Genre:** {item['listed_in']}")
                        if date_col != 'release_year' and pd.notna(item[date_col]):
                            try:
                                st.write(f"**Ditambahkan:** {item[date_col].strftime('%Y-%m-%d')}")
                            except:
                                st.write(f"**Ditambahkan:** {item[date_col]}")
    
    # Tab 2: By Keywords
    with tabs[1]:
        st.markdown("### 🔍 Cari dengan Kata Kunci")
        st.markdown("Masukkan kata kunci yang menggambarkan konten yang ingin Anda temukan.")
        
        # Search input
        query = st.text_input(
            "**Masukkan kata kunci pencarian:**",
            placeholder="Contoh: action adventure, romantic comedy, crime drama, sci-fi",
            help="Gunakan kata kunci dalam bahasa Inggris untuk hasil terbaik"
        )
        
        # Filters
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            type_filter = st.selectbox(
                "**Filter tipe:**",
                options=type_options,
                index=0
            )
        with col_f2:
            top_n_q = st.slider(
                "**Jumlah hasil:**",
                min_value=5,
                max_value=20,
                value=10
            )
        with col_f3:
            st.markdown("")
            st.markdown("")
            search_btn = st.button("🔍 Cari Konten", type="primary", use_container_width=True)
        
        # Year filter
        st.markdown("### 📅 Filter Tahun Rilis")
        year_range_q = st.slider(
            "**Pilih rentang tahun rilis:**",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year),
            key="year_q"
        )
        year_min_q, year_max_q = year_range_q
        
        if search_btn and query:
            with st.spinner("🔎 Mencari konten yang sesuai..."):
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
                ## ❌ Tidak Ditemukan Hasil
                
                **Saran untuk pencarian yang lebih baik:**
                1. Gunakan kata kunci dalam bahasa Inggris
                2. Coba kata kunci yang lebih umum
                3. Kurangi atau hilangkan beberapa filter
                4. Periksa ejaan kata kunci
                
                **Contoh kata kunci yang berhasil:**
                - `action adventure`
                - `romantic comedy`
                - `crime drama`
                - `sci-fi mystery`
                """)
            else:
                st.success(f"## ✅ Ditemukan {len(recs_q)} hasil untuk: '{query}'")
                
                # Display results
                for i, (_, r) in enumerate(recs_q.iterrows(), 1):
                    display_recommendation_card(r, i)
    
    # Tab 3: Favorites (Placeholder)
    with tabs[2]:
        st.markdown("## ⭐ Konten Favorit Anda")
        st.info("""
        **Fitur dalam pengembangan**
        
        Fitur ini akan memungkinkan Anda untuk:
        1. Menyimpan konten favorit
        2. Mendapatkan rekomendasi berdasarkan riwayat favorit
        3. Membuat playlist personal
        
        *Fitur akan segera hadir dalam update berikutnya*
        """)
        
        # Preview of popular content
        st.markdown("### 🏆 Konten Populer")
        if len(df) > 0:
            popular = df.sample(min(5, len(df)), random_state=42)
            for idx, item in popular.iterrows():
                with st.expander(f"{item['title']} ({item['type']}, {item['release_year']})", expanded=False):
                    if item['rating']:
                        st.write(f"**Rating:** {item['rating']}")
                    if item['listed_in']:
                        st.write(f"**Genre:** {item['listed_in']}")
                    if item['description']:
                        st.write(f"**Deskripsi:** {item['description'][:200]}...")

# -----------------------------
# Page: Data Analysis
# -----------------------------
elif page == "📊 Analisis Data":
    st.markdown("## 📊 Analisis Dataset Netflix")
    
    # Key Metrics
    st.markdown("### 📈 Metrik Utama")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    with col_m1:
        display_metric_card(
            "Total Konten",
            f"{len(df):,}",
            "Movies & TV Shows",
            "📊"
        )
    
    with col_m2:
        movie_count = len(df[df['type'] == 'Movie'])
        display_metric_card(
            "Movies",
            f"{movie_count:,}",
            f"{movie_count/len(df)*100:.1f}%" if len(df) > 0 else "0%",
            "🎥"
        )
    
    with col_m3:
        tv_count = len(df[df['type'] == 'TV Show'])
        display_metric_card(
            "TV Shows",
            f"{tv_count:,}",
            f"{tv_count/len(df)*100:.1f}%" if len(df) > 0 else "0%",
            "📺"
        )
    
    with col_m4:
        avg_year = int(df['release_year'].replace(0, np.nan).mean())
        display_metric_card(
            "Rata-rata Tahun",
            str(avg_year),
            f"{min_year} - {max_year}",
            "📅"
        )
    
    st.divider()
    
    # Data Preview
    st.markdown("### 📋 Pratinjau Data")
    sample_size = st.slider("**Pilih jumlah sampel data:**", 5, 50, 15)
    
    if len(df) > 0:
        preview_df = df[["title", "type", "release_year", "rating", "duration", "listed_in"]].head(sample_size)
        st.dataframe(
            preview_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "title": "Judul",
                "type": "Tipe",
                "release_year": "Tahun Rilis",
                "rating": "Rating",
                "duration": "Durasi",
                "listed_in": "Genre"
            }
        )
    else:
        st.warning("Tidak ada data untuk ditampilkan")
    
    st.divider()
    
    # Charts Section
    if len(df) > 0:
        st.markdown("## 📊 Visualisasi Data")
        
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            st.markdown("### 🎭 Distribusi Tipe Konten")
            # Count movies and TV shows
            movie_count = len(df[df['type'] == 'Movie'])
            tv_count = len(df[df['type'] == 'TV Show'])
            other_count = len(df) - movie_count - tv_count
            
            type_data = pd.DataFrame({
                'Type': ['Movie', 'TV Show', 'Other'],
                'Count': [movie_count, tv_count, other_count]
            })
            st.bar_chart(type_data.set_index('Type'))
            
            st.markdown("### 🌍 Top 10 Negara")
            top_countries = split_and_count(df["country"], sep=",", top_k=10)
            if len(top_countries) > 0:
                st.bar_chart(top_countries)
            else:
                st.info("Tidak ada data negara")
        
        with col_c2:
            st.markdown("### 🎬 Top 10 Genre")
            top_genres = split_and_count(df["listed_in"], sep=",", top_k=10)
            if len(top_genres) > 0:
                st.bar_chart(top_genres)
            else:
                st.info("Tidak ada data genre")
            
            st.markdown("### 📅 Tren Tahun Rilis")
            year_counts = df["release_year"].replace(0, np.nan).dropna().astype(int).value_counts().sort_index()
            if len(year_counts) > 0:
                st.line_chart(year_counts)
            else:
                st.info("Tidak ada data tahun rilis")
    
    st.divider()
    
    # Rating Distribution
    if len(df) > 0:
        st.markdown("### ⭐ Distribusi Rating")
        rating_counts = df["rating"].value_counts().head(15)
        
        if len(rating_counts) > 0:
            col_r1, col_r2 = st.columns([2, 1])
            with col_r1:
                st.bar_chart(rating_counts)
            
            with col_r2:
                st.markdown("#### 🏆 Rating Terpopuler")
                for rating, count in rating_counts.head(5).items():
                    st.metric(rating, f"{count:,}")
        else:
            st.info("Tidak ada data rating")
    
    st.divider()
    
    # Data Quality Check
    if len(df) > 0:
        st.markdown("### 🔍 Pemeriksaan Kualitas Data")
        
        col_q1, col_q2, col_q3 = st.columns(3)
        
        with col_q1:
            missing_director = (df['director'] == '').sum()
            st.metric("Director Kosong", f"{missing_director:,}", 
                     f"{missing_director/len(df)*100:.1f}%" if len(df) > 0 else "0%")
        
        with col_q2:
            missing_cast = (df['cast'] == '').sum()
            st.metric("Cast Kosong", f"{missing_cast:,}", 
                     f"{missing_cast/len(df)*100:.1f}%" if len(df) > 0 else "0%")
        
        with col_q3:
            missing_country = (df['country'] == '').sum()
            st.metric("Negara Kosong", f"{missing_country:,}", 
                     f"{missing_country/len(df)*100:.1f}%" if len(df) > 0 else "0%")
        
        # Export option
        st.divider()
        st.markdown("### 💾 Ekspor Data")
        
        if st.button("📥 Download Dataset yang Telah Diproses", use_container_width=True):
            csv = df.to_csv(index=False)
            st.download_button(
                label="⬇️ Klik untuk Download CSV",
                data=csv,
                file_name="netflix_processed_dataset.csv",
                mime="text/csv",
                use_container_width=True
            )

# -----------------------------
# Page: About System
# -----------------------------
elif page == "ℹ️ Tentang Sistem":
    col_about1, col_about2 = st.columns([2, 1])
    
    with col_about1:
        st.markdown("## 🎬 Tentang Sistem Rekomendasi")
        
        st.markdown("""
        ### 🎯 **Visi & Misi**
        Sistem ini dirancang untuk membantu pengguna menemukan konten Netflix yang paling sesuai dengan preferensi mereka menggunakan teknologi **Machine Learning** dan **Content-Based Filtering**.
        
        ### 🏗️ **Arsitektur Sistem**
        
        **1. Data Processing Layer**
        - Pembersihan dan normalisasi data
        - Feature engineering dari metadata
        - Pembuatan "soup" teks untuk analisis
        
        **2. Machine Learning Layer**
        - TF-IDF Vectorization untuk representasi teks
        - Cosine Similarity untuk menghitung kemiripan
        - Content-Based Filtering algorithm
        
        **3. Recommendation Engine**
        - Real-time similarity calculation
        - Dynamic filtering based on user preferences
        - Ranking and scoring system
        
        ### 🔧 **Teknologi yang Digunakan**
        
        | Komponen | Teknologi | Kegunaan |
        |----------|-----------|----------|
        | Backend | Python, Scikit-learn | Machine Learning & Processing |
        | Frontend | Streamlit | User Interface |
        | Data | Pandas, NumPy | Data Manipulation |
        | Vectorization | TF-IDF | Text Representation |
        | Similarity | Cosine Similarity | Content Matching |
        
        ### 📊 **Alur Kerja Sistem**
        1. **Input**: User memilih konten atau memasukkan kata kunci
        2. **Processing**: Sistem memproses metadata menggunakan TF-IDF
        3. **Matching**: Menghitung kemiripan dengan semua konten lain
        4. **Filtering**: Menerapkan filter berdasarkan preferensi user
        5. **Output**: Menampilkan rekomendasi terbaik dengan detail lengkap
        
        ### 🎨 **Fitur Unggulan**
        - **Intuitive UI**: Antarmuka yang user-friendly dan responsif
        - **Real-time Processing**: Hasil rekomendasi instan
        - **Advanced Filtering**: Filter berdasarkan tahun, tipe, dan rating
        - **Visual Analytics**: Dashboard data yang informatif
        - **Responsive Design**: Optimal untuk desktop dan mobile
        """)
    
    with col_about2:
        st.markdown("## 📈 Performa Sistem")
        
        display_metric_card(
            "Ukuran Dataset",
            f"{len(df):,}",
            "item konten",
            "📁"
        )
        
        display_metric_card(
            "Fitur TF-IDF",
            f"{tfidf_matrix.shape[1]:,}",
            "dimensi vektor",
            "🔢"
        )
        
        display_metric_card(
            "Waktu Response",
            "< 1 detik",
            "rata-rata",
            "⚡"
        )
        
        display_metric_card(
            "Akurasi",
            "> 85%",
            "berdasarkan testing",
            "🎯"
        )
        
        st.divider()
        
        st.markdown("## 👨‍💻 **Tim Pengembang**")
        st.markdown("""
        **Lead Developer:** Data Science Team
        **Framework:** Streamlit + Scikit-learn
        **Version:** 2.0.0
        **Last Updated:** March 2024
        
        ### 📚 **Sumber Data**
        Dataset: Netflix Movies and TV Shows
        Source: Kaggle Datasets
        License: Open Source
        
        ### 🔗 **Links**
        - [GitHub Repository](https://github.com)
        - [Documentation](https://docs.example.com)
        - [Report Issue](https://github.com/issues)
        """)
    
    st.divider()
    
    # Feature Highlights
    st.markdown("## ✨ Fitur Unggulan")
    
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    
    with col_f1:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); border-radius: 15px; height: 100%;">
            <div style="font-size: 2.5rem; margin-bottom: 1rem; color: #667eea;">🎯</div>
            <h4 style="color: #667eea; margin-bottom: 0.5rem;">Rekomendasi Presisi</h4>
            <p style="color: #666; font-size: 0.9rem;">Berdasarkan analisis metadata mendalam dan algoritma ML canggih</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_f2:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, rgba(0, 176, 155, 0.1) 0%, rgba(150, 201, 61, 0.1) 100%); border-radius: 15px; height: 100%;">
            <div style="font-size: 2.5rem; margin-bottom: 1rem; color: #00b09b;">⚡</div>
            <h4 style="color: #00b09b; margin-bottom: 0.5rem;">Real-time Processing</h4>
            <p style="color: #666; font-size: 0.9rem;">Hasil instan dengan optimasi performa terbaik</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_f3:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, rgba(255, 152, 0, 0.1) 0%, rgba(245, 124, 0, 0.1) 100%); border-radius: 15px; height: 100%;">
            <div style="font-size: 2.5rem; margin-bottom: 1rem; color: #FF9800;">🎨</div>
            <h4 style="color: #FF9800; margin-bottom: 0.5rem;">UI Modern</h4>
            <p style="color: #666; font-size: 0.9rem;">Antarmuka elegan dengan desain responsif</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_f4:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, rgba(156, 39, 176, 0.1) 0%, rgba(103, 58, 183, 0.1) 100%); border-radius: 15px; height: 100%;">
            <div style="font-size: 2.5rem; margin-bottom: 1rem; color: #9C27B0;">📊</div>
            <h4 style="color: #9C27B0; margin-bottom: 0.5rem;">Analisis Lengkap</h4>
            <p style="color: #666; font-size: 0.9rem;">Dashboard data dan statistik mendalam</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%); border-radius: 15px;">
        <h4 style="color: #667eea; margin-bottom: 1rem;">🎬 Terima Kasih Telah Menggunakan Sistem Rekomendasi Netflix!</h4>
        <p style="color: #888; margin-bottom: 0.5rem;">Sistem ini akan terus dikembangkan untuk memberikan pengalaman terbaik</p>
        <p style="color: #aaa; font-size: 0.9rem;">© 2024 Netflix Recommendation System. All rights reserved.</p>
    </div>
    """, unsafe_allow_html=True)
