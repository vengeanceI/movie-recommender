import streamlit as st
import pandas as pd
import requests
import json
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---- Data setup helper ----
try:
    from setup_data import download_tmdb_data
except Exception:
    download_tmdb_data = None

# ---- API Key Helper ----
def get_tmdb_api_key():
    """Fetch API key from environment variable or Streamlit secrets."""
    key = os.environ.get("TMDB_API_KEY")
    if key:
        return key.strip()
    try:
        key = st.secrets.get("TMDB_API_KEY")
        if key:
            return key.strip()
    except Exception:
        pass
    return None

API_KEY = get_tmdb_api_key()

# ---- Page config ----
st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# ---- Ensure dataset exists ----
if not (os.path.exists("tmdb_5000_movies.csv") and os.path.exists("tmdb_5000_credits.csv")):
    if download_tmdb_data:
        with st.spinner("📥 Downloading TMDB dataset (first run)…"):
            ok = download_tmdb_data()
            if not ok:
                st.error("❌ Could not download dataset automatically. Please try again later.")
                st.stop()
    else:
        st.error("❌ Dataset missing and setup_data not available.")
        st.stop()

# ---- Session state ----
if 'n_recs' not in st.session_state:
    st.session_state.n_recs = 5
if 'movies_df' not in st.session_state:
    st.session_state.movies_df = None
if 'similarity_matrix' not in st.session_state:
    st.session_state.similarity_matrix = None

# ---- Load dataset ----
@st.cache_data
def load_tmdb_data():
    try:
        movies = pd.read_csv('tmdb_5000_movies.csv')
        credits = pd.read_csv('tmdb_5000_credits.csv')
        movies = movies.merge(credits, left_on='id', right_on='movie_id', how='left')
        movies = movies[['id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew',
                         'vote_average', 'vote_count', 'popularity', 'release_date', 'runtime']]
        movies = movies.dropna(subset=['title', 'overview'])

        def extract_names(text, count=3):
            try:
                data = json.loads(text.replace("'", '"'))
                return ' '.join([item['name'] for item in data[:count]])
            except:
                return ''

        def extract_director(text):
            try:
                data = json.loads(text.replace("'", '"'))
                for person in data:
                    if person['job'] == 'Director':
                        return person['name']
                return ''
            except:
                return ''

        movies['genres'] = movies['genres'].fillna('[]').apply(lambda x: extract_names(x, 3))
        movies['keywords'] = movies['keywords'].fillna('[]').apply(lambda x: extract_names(x, 5))
        movies['cast'] = movies['cast'].fillna('[]').apply(lambda x: extract_names(x, 3))
        movies['director'] = movies['crew'].fillna('[]').apply(extract_director)

        movies['combined_features'] = (
            movies['overview'].fillna('') + ' ' +
            movies['genres'].fillna('') + ' ' +
            movies['keywords'].fillna('') + ' ' +
            movies['cast'].fillna('') + ' ' +
            movies['director'].fillna('')
        )

        movies = movies[movies['combined_features'].str.len() > 20]
        return movies.reset_index(drop=True)
    except Exception as e:
        st.error(f"Error loading TMDB data: {e}")
        return None

# ---- Similarity matrix ----
@st.cache_data
def create_similarity_matrix(movies_df):
    try:
        cv = CountVectorizer(max_features=5000, stop_words='english', lowercase=True)
        vectors = cv.fit_transform(movies_df['combined_features']).toarray()
        return cosine_similarity(vectors)
    except Exception as e:
        st.error(f"Error creating similarity matrix: {e}")
        return None

# ---- Posters ----
def get_movie_poster(movie_title, tmdb_id=None):
    if not API_KEY:
        return "https://via.placeholder.com/300x450/1f1f1f/ffffff?text=No+API+Key"
    try:
        if tmdb_id:
            url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={API_KEY}"
        else:
            url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie_title}"
        data = requests.get(url, timeout=5).json()
        if tmdb_id and 'poster_path' in data and data['poster_path']:
            return f"https://image.tmdb.org/t/p/w300{data['poster_path']}"
        elif 'results' in data and data['results'] and data['results'][0]['poster_path']:
            return f"https://image.tmdb.org/t/p/w300{data['results'][0]['poster_path']}"
    except:
        pass
    return "https://via.placeholder.com/300x450/1f1f1f/ffffff?text=No+Poster"

# ---- Recommendation ----
def recommend_movies(movie_title, movies_df, similarity_matrix, n_recommendations=5):
    try:
        idx = movies_df[movies_df['title'].str.lower().str.contains(movie_title.lower(), na=False)].index
        if len(idx) == 0:
            return pd.DataFrame()
        sim_scores = list(enumerate(similarity_matrix[idx[0]]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n_recommendations+1]
        recs = movies_df.iloc[[i[0] for i in sim_scores]].copy()
        recs['similarity_score'] = [s[1] for s in sim_scores]
        return recs
    except Exception as e:
        st.error(f"Error in recommendation: {e}")
        return pd.DataFrame()

# ---- Main ----
def main():
    st.title("🎬 TMDB Movie Recommendation System")
    st.markdown("*Powered by TMDB 5000 Dataset*")
    st.markdown("---")

    with st.spinner("🔄 Loading TMDB 5000 movie database..."):
        if st.session_state.movies_df is None:
            movies_df = load_tmdb_data()
            if movies_df is not None:
                st.session_state.movies_df = movies_df
                with st.spinner("🧮 Computing movie similarities..."):
                    st.session_state.similarity_matrix = create_similarity_matrix(movies_df)
            else:
                st.error("❌ Failed to load dataset.")
                return

    movies_df = st.session_state.movies_df
    similarity_matrix = st.session_state.similarity_matrix
    if movies_df is None or similarity_matrix is None:
        st.error("❌ Data load failed. Refresh and try again.")
        return

    # Sidebar
    with st.sidebar:
        st.header("🎛️ Settings")
        st.session_state.n_recs = st.slider("Number of recommendations", 1, 20, st.session_state.n_recs)
        st.markdown("---")
        st.info(f"📽️ Total movies: {len(movies_df):,}")
        st.info(f"⭐ Avg rating: {movies_df['vote_average'].mean():.1f}/10")
        st.info(f"📅 Years: {movies_df['release_date'].str[:4].min()} - {movies_df['release_date'].str[:4].max()}")

    # Movie selection
    col1, col2 = st.columns([1, 2])
    with col1:
        search_query = st.text_input("🔎 Search for a movie:", placeholder="Type movie name...")
        if search_query:
            options = movies_df[movies_df['title'].str.lower().str.contains(search_query.lower())]['title'].tolist()[:20]
        else:
            options = movies_df.nlargest(50, 'popularity')['title'].tolist()
        selected_movie = st.selectbox("Choose a movie:", options) if options else None

        if selected_movie:
            info = movies_df[movies_df['title'] == selected_movie].iloc[0]
            st.markdown("### 📋 Movie Details")
            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.image(get_movie_poster(selected_movie, info.get('id')), width=150)
            with col_b:
                st.write(f"⭐ {info['vote_average']:.1f}/10 ({info['vote_count']:,} votes)")
                st.write(f"📅 {info['release_date'][:4] if pd.notna(info['release_date']) else 'Unknown'}")
                if pd.notna(info['runtime']):
                    st.write(f"⏱️ {int(info['runtime'])} min")
            st.write(f"🎭 {info['genres']}")
            st.write(f"👥 {info['cast']}")
            st.write(f"🎬 {info['director']}")
            with st.expander("📝 Plot Summary"):
                st.write(info['overview'])

    with col2:
        if selected_movie:
            st.header("🎯 Recommended Movies")
            recs = recommend_movies(selected_movie, movies_df, similarity_matrix, st.session_state.n_recs)
            if not recs.empty:
                for i, (_, movie) in enumerate(recs.iterrows(), 1):
                    with st.container():
                        c1, c2, c3 = st.columns([1, 2, 1])
                        with c1:
                            st.image(get_movie_poster(movie['title'], movie.get('id')), width=100)
                        with c2:
                            st.markdown(f"### {i}. {movie['title']}")
                            st.write(f"⭐ {movie['vote_average']:.1f}/10")
                            st.write(f"📅 {movie['release_date'][:4] if pd.notna(movie['release_date']) else 'Unknown'}")
                            st.write(f"🎭 {movie['genres']}")
                        with c3:
                            st.metric("Match", f"{movie['similarity_score']*100:.0f}%")
                        st.markdown("---")

if __name__ == "__main__":
    main()
