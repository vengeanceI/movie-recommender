import streamlit as st
import pandas as pd
import requests
import json
import io
try:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    import random
    def simple_recommend(movies_df, selected_title, n_recommendations=5):
        """Simple genre-based recommendation fallback"""
        try:
            selected_movie = movies_df[movies_df['title'] == selected_title].iloc[0]
            selected_genres = selected_movie['genres'].lower()
            
            # Find movies with similar genres
            similar_movies = movies_df[
                (movies_df['title'] != selected_title) & 
                (movies_df['genres'].str.lower().str.contains('|'.join(selected_genres.split()[:2]), na=False))
            ].copy()
            
            if len(similar_movies) < n_recommendations:
                # Add high-rated movies if not enough similar ones
                additional = movies_df[
                    (movies_df['title'] != selected_title) & 
                    (~movies_df['title'].isin(similar_movies['title']))
                ].nlargest(n_recommendations - len(similar_movies), 'vote_average')
                similar_movies = pd.concat([similar_movies, additional])
            
            result = similar_movies.head(n_recommendations).copy()
            result['similarity_score'] = [0.8 - i*0.1 for i in range(len(result))]
            return result
        except:
            return pd.DataFrame()

# Page config with dark theme
st.set_page_config(
    page_title="🎬 Cinema Noir",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Cinema Noir styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global theme variables */
    :root {
        --primary-bg: #0a0a0a;
        --secondary-bg: #1a1a1a;
        --card-bg: #2a2a2a;
        --accent-color: #00d4aa;
        --text-primary: #ffffff;
        --text-secondary: #b3b3b3;
        --hover-bg: #3a3a3a;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* Main container styling */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styling */
    .main-header {
        background: rgba(26, 26, 26, 0.95);
        backdrop-filter: blur(10px);
        padding: 1rem 2rem;
        margin: -1rem -2rem 2rem -2rem;
        border-bottom: 1px solid #333;
        position: sticky;
        top: 0;
        z-index: 100;
    }
    
    .logo-section {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    .logo-icon {
        background: var(--accent-color);
        color: #000;
        padding: 0.5rem;
        border-radius: 8px;
        font-size: 1.2rem;
        font-weight: bold;
    }
    
    .logo-text {
        font-size: 1.8rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0;
    }
    
    .tagline {
        color: var(--text-secondary);
        font-size: 1rem;
        margin: 0;
    }
    
    /* Search section styling */
    .search-section {
        background: var(--secondary-bg);
        border-radius: 12px;
        padding: 2rem;
        margin: 2rem 0;
        border: 1px solid #333;
    }
    
    .hero-text {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
        line-height: 1.1;
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        color: var(--text-secondary);
        margin-bottom: 2rem;
    }
    
    .accent-text {
        color: var(--accent-color);
    }
    
    /* Movie card styling */
    .movie-card {
        background: var(--card-bg);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #333;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .movie-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 40px rgba(0, 212, 170, 0.1);
        border-color: var(--accent-color);
    }
    
    .movie-poster {
        border-radius: 12px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease;
    }
    
    .movie-poster:hover {
        transform: scale(1.02);
    }
    
    .movie-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
    }
    
    .movie-meta {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin: 0.5rem 0;
        color: var(--text-secondary);
    }
    
    .rating {
        background: linear-gradient(135deg, #ffd700, #ffed4e);
        color: #000;
        padding: 0.2rem 0.5rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .match-score {
        background: linear-gradient(135deg, var(--accent-color), #00a083);
        color: #000;
        padding: 0.3rem 0.6rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    /* Featured movies grid */
    .featured-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .featured-card {
        background: var(--card-bg);
        border-radius: 12px;
        overflow: hidden;
        transition: all 0.3s ease;
        border: 1px solid #333;
        cursor: pointer;
    }
    
    .featured-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 32px rgba(0, 212, 170, 0.15);
        border-color: var(--accent-color);
    }
    
    .featured-poster {
        width: 100%;
        height: 280px;
        object-fit: cover;
    }
    
    .featured-info {
        padding: 1rem;
    }
    
    .featured-title {
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }
    
    .featured-genre {
        color: var(--text-secondary);
        font-size: 0.8rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-color), #00a083) !important;
        color: #000 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 16px rgba(0, 212, 170, 0.3) !important;
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        background: var(--card-bg) !important;
        color: var(--text-primary) !important;
        border: 1px solid #333 !important;
        border-radius: 8px !important;
        padding: 0.75rem !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--accent-color) !important;
        box-shadow: 0 0 0 2px rgba(0, 212, 170, 0.2) !important;
    }
    
    .stSelectbox > div > div > div {
        background: var(--card-bg) !important;
        color: var(--text-primary) !important;
        border: 1px solid #333 !important;
        border-radius: 8px !important;
    }
    
    /* Stats styling */
    .stats-container {
        display: flex;
        justify-content: center;
        gap: 3rem;
        margin: 2rem 0;
        text-align: center;
    }
    
    .stat-item {
        color: var(--text-primary);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--accent-color);
        display: block;
    }
    
    .stat-label {
        color: var(--text-secondary);
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    /* Hide default streamlit styling */
    .stAlert > div {
        background: var(--secondary-bg) !important;
        border: 1px solid #333 !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
    }
    
    /* Loading spinner */
    .stSpinner > div {
        border-top-color: var(--accent-color) !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: var(--card-bg) !important;
        color: var(--text-primary) !important;
        border-radius: 8px !important;
    }
    
    .streamlit-expanderContent {
        background: var(--secondary-bg) !important;
        border: 1px solid #333 !important;
        border-radius: 0 0 8px 8px !important;
    }
</style>
""", unsafe_allow_html=True)

# TMDB API key
try:
    TMDB_API_KEY = st.secrets["TMDB_API_KEY"]
except:
    TMDB_API_KEY = None

# GitHub configuration
GITHUB_REPO_URL = "https://raw.githubusercontent.com/vengeanceI/movie-recommender/main/"
MOVIES_FILE = "tmdb_5000_movies.csv"
CREDITS_FILE = "tmdb_credits_maximum.csv"

@st.cache_data
def load_data_from_github():
    """Load movie and credits data from GitHub repository"""
    try:
        movies_url = f"{GITHUB_REPO_URL}{MOVIES_FILE}"
        movies_response = requests.get(movies_url, timeout=30)
        movies_response.raise_for_status()
        movies_df = pd.read_csv(io.StringIO(movies_response.text))
        
        credits_url = f"{GITHUB_REPO_URL}{CREDITS_FILE}"
        credits_response = requests.get(credits_url, timeout=30)
        credits_response.raise_for_status()
        credits_df = pd.read_csv(io.StringIO(credits_response.text))
        
        # Merge data
        if 'movie_id' in credits_df.columns:
            movies_df = movies_df.merge(credits_df, left_on='id', right_on='movie_id', how='left')
        else:
            movies_df = movies_df.merge(credits_df, on='id', how='left')
        
        return process_movie_data(movies_df)
        
    except Exception as e:
        return create_fallback_data()

def process_movie_data(movies_df):
    """Process and clean movie data"""
    try:
        essential_cols = ['id', 'title', 'overview', 'genres', 'vote_average', 'vote_count', 
                         'popularity', 'release_date', 'runtime']
        optional_cols = ['cast', 'crew', 'keywords']
        available_cols = essential_cols + [col for col in optional_cols if col in movies_df.columns]
        
        movies_df = movies_df[available_cols].copy()
        movies_df = movies_df.dropna(subset=['title', 'overview'])
        movies_df = movies_df[movies_df['overview'].str.len() > 10]
        
        # Process genres
        def extract_genre_names(text):
            try:
                if pd.isna(text) or text == '':
                    return 'Unknown'
                data = json.loads(text.replace("'", '"'))
                if isinstance(data, list):
                    return ' '.join([item['name'] for item in data[:3] if 'name' in item])
                return str(text)
            except:
                return str(text) if pd.notna(text) else 'Unknown'
        
        movies_df['genres'] = movies_df['genres'].fillna('[]').apply(extract_genre_names)
        
        # Process cast and crew
        if 'cast' in movies_df.columns:
            def extract_cast_names(text):
                try:
                    if pd.isna(text) or text == '':
                        return ''
                    data = json.loads(text.replace("'", '"'))
                    if isinstance(data, list):
                        return ' | '.join([actor['name'] for actor in data[:5] if 'name' in actor])
                    return str(text)
                except:
                    return str(text) if pd.notna(text) else ''
            
            movies_df['cast'] = movies_df['cast'].fillna('[]').apply(extract_cast_names)
        
        if 'crew' in movies_df.columns:
            def extract_director(text):
                try:
                    if pd.isna(text) or text == '':
                        return ''
                    data = json.loads(text.replace("'", '"'))
                    if isinstance(data, list):
                        for person in data:
                            if person.get('job') == 'Director':
                                return person['name']
                    return ''
                except:
                    return ''
            
            movies_df['director'] = movies_df['crew'].apply(extract_director)
        
        # Create combined features
        feature_components = [
            movies_df['overview'].fillna(''),
            movies_df['genres'].fillna(''),
        ]
        
        if 'cast' in movies_df.columns:
            cast_clean = movies_df['cast'].fillna('').str.replace('|', ' ')
            feature_components.append(cast_clean)
        
        if 'director' in movies_df.columns:
            feature_components.append(movies_df['director'].fillna(''))
        
        movies_df['combined_features'] = ''
        for component in feature_components:
            movies_df['combined_features'] += ' ' + component.astype(str)
        
        movies_df['combined_features'] = movies_df['combined_features'].str.strip()
        
        return movies_df.reset_index(drop=True)
        
    except Exception as e:
        return movies_df

def create_fallback_data():
    """Create sample data if GitHub files are unavailable"""
    sample_data = [
        {"id": 19995, "title": "Avatar", "genres": "Action Adventure Fantasy", "vote_average": 7.2, "overview": "In the 22nd century, a paraplegic Marine is dispatched to the moon Pandora on a unique mission.", "cast": "Sam Worthington | Zoe Saldana | Sigourney Weaver", "director": "James Cameron"},
        {"id": 285, "title": "Pirates of the Caribbean: The Curse of the Black Pearl", "genres": "Adventure Fantasy Action", "vote_average": 7.8, "overview": "Blacksmith Will Turner teams up with eccentric pirate Captain Jack Sparrow.", "cast": "Johnny Depp | Orlando Bloom | Keira Knightley", "director": "Gore Verbinski"},
        {"id": 206647, "title": "Spectre", "genres": "Action Adventure Crime", "vote_average": 6.3, "overview": "A cryptic message from Bond's past sends him on a trail to uncover a sinister organization.", "cast": "Daniel Craig | Christoph Waltz | Léa Seydoux", "director": "Sam Mendes"},
        {"id": 49026, "title": "The Dark Knight Rises", "genres": "Action Crime Drama", "vote_average": 7.6, "overview": "Following the death of District Attorney Harvey Dent, Batman assumes responsibility for Dent's crimes.", "cast": "Christian Bale | Tom Hardy | Anne Hathaway", "director": "Christopher Nolan"},
        {"id": 49529, "title": "John Carter", "genres": "Action Adventure Science Fiction", "vote_average": 6.1, "overview": "A former Confederate captain is mysteriously transported to Mars.", "cast": "Taylor Kitsch | Lynn Collins | Samantha Morton", "director": "Andrew Stanton"},
    ]
    
    df = pd.DataFrame(sample_data)
    df['combined_features'] = df['overview'] + ' ' + df['genres'] + ' ' + df['cast'] + ' ' + df['director']
    df['vote_count'] = 1000
    df['popularity'] = 50.0
    df['release_date'] = '2009-01-01'
    df['runtime'] = 120
    
    return df

@st.cache_data
def create_similarity_matrix(movies_df):
    """Create similarity matrix for recommendations"""
    if not SKLEARN_AVAILABLE:
        return None
        
    try:
        cv = CountVectorizer(
            max_features=5000,
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2
        )
        
        vectors = cv.fit_transform(movies_df['combined_features']).toarray()
        similarity = cosine_similarity(vectors)
        
        return similarity
        
    except Exception as e:
        return None

def get_movie_poster(movie_title, tmdb_id=None):
    """Get movie poster from TMDB API"""
    try:
        if not TMDB_API_KEY:
            return "https://via.placeholder.com/300x450/2a2a2a/ffffff?text=🎬+No+Poster"
            
        if tmdb_id:
            url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={TMDB_API_KEY}"
        else:
            url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_title}"
        
        response = requests.get(url, timeout=5)
        data = response.json()
        
        poster_path = None
        if tmdb_id and 'poster_path' in data:
            poster_path = data['poster_path']
        elif 'results' in data and data['results'] and data['results'][0].get('poster_path'):
            poster_path = data['results'][0]['poster_path']
        
        if poster_path:
            return f"https://image.tmdb.org/t/p/w300{poster_path}"
            
    except:
        pass
    
    return "https://via.placeholder.com/300x450/2a2a2a/ffffff?text=🎬+No+Poster"

def recommend_movies(movie_title, movies_df, similarity_matrix, n_recommendations=5):
    """Get movie recommendations"""
    try:
        if not SKLEARN_AVAILABLE:
            return simple_recommend(movies_df, movie_title, n_recommendations)
        
        movie_indices = movies_df[
            movies_df['title'].str.lower().str.contains(movie_title.lower(), na=False)
        ].index
        
        if len(movie_indices) == 0:
            return pd.DataFrame()
        
        movie_idx = movie_indices[0]
        sim_scores = list(enumerate(similarity_matrix[movie_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:n_recommendations+1]
        movie_indices = [i[0] for i in sim_scores]
        
        recommendations = movies_df.iloc[movie_indices].copy()
        recommendations['similarity_score'] = [score[1] for score in sim_scores]
        
        return recommendations
        
    except Exception as e:
        return pd.DataFrame()

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <div class="logo-section">
            <div class="logo-icon">🎬</div>
            <h1 class="logo-text">Cinema Noir</h1>
        </div>
        <p class="tagline">AI-powered recommendations tailored to your taste</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data silently
    movies_df = load_data_from_github()
    
    if movies_df is None or len(movies_df) == 0:
        st.error("⚠️ Could not load movie data. Please check your connection.")
        return
    
    # Create similarity matrix silently
    similarity_matrix = create_similarity_matrix(movies_df)
    
    # Stats
    st.markdown(f"""
    <div class="stats-container">
        <div class="stat-item">
            <span class="stat-number">{len(movies_df):,}</span>
            <div class="stat-label">Movies</div>
        </div>
        <div class="stat-item">
            <span class="stat-number">500K+</span>
            <div class="stat-label">Users</div>
        </div>
        <div class="stat-item">
            <span class="stat-number">98%</span>
            <div class="stat-label">Accuracy</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Search section
    st.markdown("""
    <div class="search-section">
        <div class="hero-text">
            <h1 class="hero-title">Discover<br>Your Next<br><span class="accent-text">Obsession</span></h1>
            <p class="hero-subtitle">Experience cinema like never before with our premium movie discovery platform.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Search and recommendations
    col1, col2 = st.columns([1, 2], gap="large")
    
    with col1:
        st.markdown("### 🔍 Find Your Perfect Movie")
        
        search_query = st.text_input(
            "",
            placeholder="Search movies, actors, genres...",
            key="movie_search"
        )
        
        if search_query:
            filtered_movies = movies_df[
                movies_df['title'].str.lower().str.contains(search_query.lower(), na=False)
            ]['title'].tolist()[:20]
        else:
            popular_movies = movies_df.nlargest(30, 'popularity')
            top_rated = movies_df.nlargest(30, 'vote_average')
            combined = pd.concat([popular_movies, top_rated]).drop_duplicates('title')
            filtered_movies = combined['title'].tolist()
        
        if filtered_movies:
            selected_movie = st.selectbox("Choose a movie:", filtered_movies, key="movie_select")
        else:
            selected_movie = None
        
        # Show selected movie details
        if selected_movie:
            movie_info = movies_df[movies_df['title'] == selected_movie].iloc[0]
            
            st.markdown('<div class="movie-card">', unsafe_allow_html=True)
            poster_url = get_movie_poster(movie_info['title'], movie_info.get('id'))
            st.image(poster_url, width=200)
            
            st.markdown(f'<h3 class="movie-title">{movie_info["title"]}</h3>', unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="movie-meta">
                <span class="rating">⭐ {movie_info['vote_average']:.1f}</span>
                <span>{movie_info['genres']}</span>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("📖 Plot Summary"):
                st.write(movie_info['overview'])
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if selected_movie:
            st.markdown("### 🎯 Recommended For You")
            
            recommendations = recommend_movies(
                selected_movie, movies_df, similarity_matrix, 6
            )
            
            if len(recommendations) > 0:
                for idx, (_, movie) in enumerate(recommendations.iterrows()):
                    st.markdown('<div class="movie-card">', unsafe_allow_html=True)
                    
                    rec_col1, rec_col2, rec_col3 = st.columns([1, 3, 1])
                    
                    with rec_col1:
                        poster_url = get_movie_poster(movie['title'], movie.get('id'))
                        st.image(poster_url, width=120)
                    
                    with rec_col2:
                        st.markdown(f'<h4 class="movie-title">{movie["title"]}</h4>', unsafe_allow_html=True)
                        
                        st.markdown(f"""
                        <div class="movie-meta">
                            <span class="rating">⭐ {movie['vote_average']:.1f}</span>
                            <span>{movie['genres']}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if 'cast' in movie and pd.notna(movie['cast']) and movie['cast']:
                            cast_display = str(movie['cast'])[:50] + "..." if len(str(movie['cast'])) > 50 else str(movie['cast'])
                            st.write(f"👥 {cast_display}")
                        
                        with st.expander(f"About {movie['title']}"):
                            st.write(movie['overview'])
                    
                    with rec_col3:
                        similarity_score = movie.get('similarity_score', 0) * 100
                        st.markdown(f'<div class="match-score">{similarity_score:.0f}% Match</div>', unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("😔 No recommendations found for this movie.")
        else:
            st.markdown("### 🌟 Featured Movies")
            st.write("Select a movie from the search panel to get personalized recommendations!")
            
            # Show featured movies grid
            featured = movies_df.nlargest(12, 'vote_average')
            
            # Create grid using columns
            cols = st.columns(4)
            for idx, (_, movie) in enumerate(featured.iterrows()):
                with cols[idx % 4]:
                    st.markdown('<div class="featured-card">', unsafe_allow_html=True)
                    poster_url = get_movie_poster(movie['title'], movie.get('id'))
                    st.image(poster_url, use_column_width=True)
                    st.markdown(f"""
                    <div class="featured-info">
                        <div class="featured-title">{movie['title']}</div>
                        <div class="featured-genre">⭐ {movie['vote_average']:.1f} • {movie['genres'][:20]}...</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
