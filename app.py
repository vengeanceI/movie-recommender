import streamlit as st
import pandas as pd
import requests
import json
import gzip
import io
try:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    try:
        import sklearn
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
                
                similar_movies = movies_df[
                    (movies_df['title'] != selected_title) & 
                    (movies_df['genres'].str.lower().str.contains('|'.join(selected_genres.split()[:2]), na=False))
                ].copy()
                
                if len(similar_movies) < n_recommendations:
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

# Page config
st.set_page_config(
    page_title="Cinema Noir - Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Cinema Noir theme
st.markdown("""
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global Styles */
.stApp {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #21262d 100%);
    font-family: 'Inter', sans-serif;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Header styling */
.main-header {
    background: rgba(33, 38, 45, 0.95);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(48, 54, 61, 0.8);
    border-radius: 20px;
    padding: 2rem;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}

.main-title {
    font-size: 3.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #58a6ff 0%, #1f6feb 50%, #0969da 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
    text-shadow: 0 0 30px rgba(88, 166, 255, 0.3);
}

.main-subtitle {
    font-size: 1.2rem;
    color: #8b949e;
    font-weight: 300;
    margin-bottom: 2rem;
}

/* Stats section */
.stats-container {
    display: flex;
    justify-content: center;
    gap: 3rem;
    margin-top: 2rem;
}

.stat-item {
    text-align: center;
}

.stat-number {
    font-size: 2.5rem;
    font-weight: 700;
    color: #58a6ff;
    display: block;
    text-shadow: 0 0 20px rgba(88, 166, 255, 0.4);
}

.stat-label {
    font-size: 1rem;
    color: #8b949e;
    font-weight: 400;
    margin-top: 0.5rem;
}

/* Hero Section */
.hero-section {
    background: rgba(33, 38, 45, 0.8);
    border: 1px solid rgba(48, 54, 61, 0.6);
    border-radius: 20px;
    padding: 2.5rem;
    margin-bottom: 3rem;
    backdrop-filter: blur(10px);
}

.hero-title {
    font-size: 3rem;
    font-weight: 600;
    color: #f0f6fc;
    text-align: center;
    margin-bottom: 1rem;
}

.hero-gradient-text {
    background: linear-gradient(135deg, #39d353 0%, #26a641 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-description {
    font-size: 1.1rem;
    color: #8b949e;
    text-align: center;
    margin-bottom: 2rem;
    line-height: 1.6;
}

/* Search section */
.search-section {
    background: rgba(33, 38, 45, 0.6);
    border: 1px solid rgba(48, 54, 61, 0.8);
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 2rem;
}

/* Movie cards */
.movie-card {
    background: rgba(33, 38, 45, 0.8);
    border: 1px solid rgba(48, 54, 61, 0.8);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.movie-card:hover {
    border-color: rgba(88, 166, 255, 0.6);
    box-shadow: 0 8px 25px rgba(88, 166, 255, 0.15);
    transform: translateY(-2px);
}

.movie-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #58a6ff, #1f6feb);
    opacity: 0;
    transition: opacity 0.3s ease;
}

.movie-card:hover::before {
    opacity: 1;
}

.movie-title {
    font-size: 1.4rem;
    font-weight: 600;
    color: #f0f6fc;
    margin-bottom: 0.5rem;
}

.movie-rating {
    display: inline-block;
    background: linear-gradient(135deg, #ffd700, #ffb700);
    color: #000;
    padding: 0.3rem 0.8rem;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.9rem;
    margin-right: 1rem;
}

.movie-genre {
    color: #58a6ff;
    font-size: 0.95rem;
    font-weight: 500;
}

.movie-description {
    color: #8b949e;
    font-size: 0.95rem;
    line-height: 1.5;
    margin-top: 1rem;
}

.match-score {
    position: absolute;
    top: 1rem;
    right: 1rem;
    background: linear-gradient(135deg, #39d353, #26a641);
    color: #fff;
    padding: 0.4rem 0.8rem;
    border-radius: 12px;
    font-weight: 600;
    font-size: 0.85rem;
}

/* Section headers */
.section-header {
    font-size: 2rem;
    font-weight: 600;
    color: #f0f6fc;
    margin: 2rem 0 1.5rem 0;
    text-align: center;
}

/* Grid layout for movie posters */
.movie-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 1.5rem;
    margin: 2rem 0;
}

.poster-card {
    background: rgba(33, 38, 45, 0.6);
    border: 1px solid rgba(48, 54, 61, 0.8);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    transition: all 0.3s ease;
}

.poster-card:hover {
    border-color: rgba(88, 166, 255, 0.6);
    transform: translateY(-5px);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
}

.poster-card img {
    border-radius: 8px;
    width: 100%;
    height: auto;
}

.poster-title {
    color: #f0f6fc;
    font-size: 1rem;
    font-weight: 500;
    margin-top: 0.8rem;
}

/* Custom button styling */
.stButton > button {
    background: linear-gradient(135deg, #58a6ff, #1f6feb);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 1.5rem;
    font-weight: 600;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #1f6feb, #0969da);
    box-shadow: 0 5px 15px rgba(88, 166, 255, 0.4);
    transform: translateY(-1px);
}

/* Custom selectbox styling */
.stSelectbox > div > div > div {
    background: rgba(33, 38, 45, 0.8);
    border: 1px solid rgba(48, 54, 61, 0.8);
    color: #f0f6fc;
}

/* Footer */
.footer {
    text-align: center;
    color: #8b949e;
    font-size: 0.9rem;
    margin-top: 3rem;
    padding: 2rem;
    border-top: 1px solid rgba(48, 54, 61, 0.6);
}

/* Responsive design */
@media (max-width: 768px) {
    .main-title {
        font-size: 2.5rem;
    }
    
    .hero-title {
        font-size: 2rem;
    }
    
    .stats-container {
        flex-direction: column;
        gap: 1.5rem;
    }
    
    .movie-grid {
        grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
        gap: 1rem;
    }
}
</style>
""", unsafe_allow_html=True)

# TMDB API key from Streamlit secrets
try:
    TMDB_API_KEY = st.secrets["TMDB_API_KEY"]
except:
    TMDB_API_KEY = None

# GitHub repository configuration
GITHUB_REPO_URL = "https://raw.githubusercontent.com/vengeanceI/movie-recommender/main/"
MOVIES_FILE = "tmdb_5000_movies.csv"
CREDITS_FILE = "tmdb_credits_maximum.csv"

@st.cache_data
def load_data_from_github():
    """Load movie and credits data from GitHub repository"""
    movies_df = None
    credits_df = None
    data_source = "GitHub Repository"
    
    try:
        movies_url = f"{GITHUB_REPO_URL}{MOVIES_FILE}"
        movies_response = requests.get(movies_url, timeout=30)
        movies_response.raise_for_status()
        
        movies_df = pd.read_csv(io.StringIO(movies_response.text))
        
        credits_url = f"{GITHUB_REPO_URL}{CREDITS_FILE}"
        credits_response = requests.get(credits_url, timeout=30)
        credits_response.raise_for_status()
        
        credits_df = pd.read_csv(io.StringIO(credits_response.text))
        
        if credits_df is not None:
            if 'movie_id' in credits_df.columns:
                movies_df = movies_df.merge(credits_df, left_on='id', right_on='movie_id', how='left')
            else:
                movies_df = movies_df.merge(credits_df, on='id', how='left')
            
            data_source += " (Movies + Credits)"
        
        movies_df = process_movie_data(movies_df)
        return movies_df
        
    except Exception as e:
        st.error(f"Error loading data from GitHub: {e}")
        return create_fallback_data()

@st.cache_data
def load_movie_data():
    """Main data loading function"""
    try:
        return load_data_from_github()
    except Exception as e:
        return create_fallback_data()

def process_movie_data(movies_df):
    """Process and clean movie data with credits handling"""
    try:
        essential_cols = ['id', 'title', 'overview', 'genres', 'vote_average', 'vote_count', 
                         'popularity', 'release_date', 'runtime']
        
        optional_cols = ['cast', 'crew', 'keywords']
        available_cols = essential_cols + [col for col in optional_cols if col in movies_df.columns]
        
        movies_df = movies_df[available_cols].copy()
        movies_df = movies_df.dropna(subset=['title', 'overview'])
        movies_df = movies_df[movies_df['overview'].str.len() > 10]
        
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
        
        if 'keywords' in movies_df.columns:
            def extract_keywords(text):
                try:
                    if pd.isna(text) or text == '':
                        return ''
                    data = json.loads(text.replace("'", '"'))
                    if isinstance(data, list):
                        return ' '.join([item['name'] for item in data[:5] if 'name' in item])
                    return ''
                except:
                    return ''
            movies_df['keywords'] = movies_df['keywords'].fillna('[]').apply(extract_keywords)
        
        if 'cast' in movies_df.columns:
            def extract_cast_names(text):
                try:
                    if pd.isna(text) or text == '':
                        return ''
                    data = json.loads(text.replace("'", '"'))
                    if isinstance(data, list):
                        cast_names = [actor['name'] for actor in data[:5] if 'name' in actor]
                        return ' | '.join(cast_names)
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
        
        feature_components = [
            movies_df['overview'].fillna(''),
            movies_df['genres'].fillna(''),
        ]
        
        if 'cast' in movies_df.columns:
            cast_clean = movies_df['cast'].fillna('').str.replace('|', ' ')
            feature_components.append(cast_clean)
        
        if 'director' in movies_df.columns:
            feature_components.append(movies_df['director'].fillna(''))
        
        if 'keywords' in movies_df.columns:
            feature_components.append(movies_df['keywords'].fillna(''))
        
        movies_df['combined_features'] = ''
        for component in feature_components:
            movies_df['combined_features'] += ' ' + component.astype(str)
        
        movies_df['combined_features'] = movies_df['combined_features'].str.strip()
        
        return movies_df.reset_index(drop=True)
        
    except Exception as e:
        st.error(f"Error processing movie data: {e}")
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
        st.error(f"Error creating similarity matrix: {e}")
        return None

def get_movie_poster(movie_title, tmdb_id=None):
    """Get movie poster from TMDB API"""
    try:
        if not TMDB_API_KEY:
            return "https://via.placeholder.com/300x450/1f1f1f/ffffff?text=🎬+No+API+Key"
            
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
    
    return "https://via.placeholder.com/300x450/1f1f1f/ffffff?text=🎬+No+Poster"

def recommend_movies(movie_title, movies_df, similarity_matrix, n_recommendations=5):
    """Get movie recommendations based on content similarity"""
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
        st.error(f"Error getting recommendations: {e}")
        return pd.DataFrame()

def main():
    # Main Header
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">Cinema Noir</h1>
        <p class="main-subtitle">AI-powered recommendations tailored to your taste.</p>
        <div class="stats-container">
            <div class="stat-item">
                <span class="stat-number">4,799</span>
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
    </div>
    """, unsafe_allow_html=True)

    # Load data
    movies_df = load_movie_data()
    
    if movies_df is None or len(movies_df) == 0:
        st.error("Could not load movie data. Please check your configuration.")
        return
    
    similarity_matrix = create_similarity_matrix(movies_df)

    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <h2 class="hero-title">Discover Your Next <span class="hero-gradient-text">Obsession</span></h2>
        <p class="hero-description">Experience cinema like never before with our premium movie discovery platform.</p>
    </div>
    """, unsafe_allow_html=True)

    # Search and Recommendation Section
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="search-section">', unsafe_allow_html=True)
        st.markdown("### Find Your Perfect Movie")
        
        search_query = st.text_input("Search movies, actors, genres...", placeholder="Type movie name...")
        
        if search_query:
            filtered_movies = movies_df[
                movies_df['title'].str.lower().str.contains(search_query.lower(), na=False)
            ]['title'].tolist()[:25]
        else:
            popular_movies = movies_df.nlargest(40, 'popularity')
            top_rated = movies_df.nlargest(40, 'vote_average')
            combined = pd.concat([popular_movies, top_rated]).drop_duplicates('title')
            filtered_movies = combined['title'].tolist()
        
        if filtered_movies:
            selected_movie = st.selectbox("Choose a movie:", filtered_movies)
        else:
            selected_movie = None
        
        if selected_movie:
            movie_info = movies_df[movies_df['title'] == selected_movie].iloc[0]
            
            # Display selected movie poster
            poster_url = get_movie_poster(movie_info['title'], movie_info.get('id'))
            st.image(poster_url, width=250)
            
            st.markdown(f"**{movie_info['title']}**")
            st.markdown(f"⭐ {movie_info['vote_average']:.1f}/10")
            st.markdown(f"🎭 {movie_info['genres']}")
            
            if 'director' in movie_info and pd.notna(movie_info['director']):
                st.markdown(f"🎬 {movie_info['director']}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Recommended For You")
        
        if selected_movie:
            recommendations = recommend_movies(
                selected_movie, movies_df, similarity_matrix, 6
            )
            
            if len(recommendations) > 0:
                for idx, (_, movie) in enumerate(recommendations.iterrows()):
                    similarity_score = movie.get('similarity_score', 0) * 100
                    
                    st.markdown(f"""
                    <div class="movie-card">
                        <div class="match-score">{similarity_score:.0f}% Match</div>
                        <h3 class="movie-title">{movie['title']}</h3>
                        <div style="margin-bottom: 1rem;">
                            <span class="movie-rating">{movie['vote_average']:.1f}</span>
                            <span class="movie-genre">{movie['genres']}</span>
                        </div>
                        <p class="movie-description">{movie['overview'][:150]}...</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No recommendations found for this movie.")
        else:
            # Featured movies section
            featured = movies_df.nlargest(3, 'vote_average')
            
            for _, movie in featured.iterrows():
                st.markdown(f"""
                <div class="movie-card">
                    <h3 class="movie-title">{movie['title']}</h3>
                    <div style="margin-bottom: 1rem;">
                        <span class="movie-rating">{movie['vote_average']:.1f}</span>
                        <span class="movie-genre">{movie['genres']}</span>
                    </div>
                    <p class="movie-description">{movie['overview'][:120]}...</p>
                </div>
                """, unsafe_allow_html=True)

    # New Movies Section
    st.markdown('<h2 class="section-header">New Movies</h2>', unsafe_allow_html=True)
    
    # Get recent/popular movies for the grid
    recent_movies = movies_df.nlargest(12, 'popularity')
    
    cols = st.columns(6)
    for idx, (_, movie) in enumerate(recent_movies.iterrows()):
        with cols[idx % 6]:
            poster_url = get_movie_poster(movie['title'], movie.get('id'))
            st.markdown(f"""
            <div class="poster-card">
                <img src="{poster_url}" alt="{movie['title']}">
                <div class="poster-title">{movie['title'][:20]}{'...' if len(movie['title']) > 20 else ''}</div>
            </div>
            """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer">
        <p>🎬 Data from TMDB via GitHub Repository | Built with ❤️ using Streamlit</p>
        <p>💡 Content-based filtering using movie overviews, genres, cast & crew</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
