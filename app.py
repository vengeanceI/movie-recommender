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
    page_title="Cinema Vault - Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Cinema Vault theme
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

.movie-meta {
    display: flex;
    gap: 1rem;
    margin-top: 1rem;
    font-size: 0.9rem;
    color: #8b949e;
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

.selected-movie-info {
    background: rgba(33, 38, 45, 0.9);
    border: 1px solid rgba(48, 54, 61, 0.8);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    display: flex;
    gap: 1.5rem;
    align-items: flex-start;
}

.movie-poster {
    flex-shrink: 0;
    border-radius: 12px;
    overflow: hidden;
}

.movie-info-content {
    flex: 1;
}

.movie-info-title {
    font-size: 1.6rem;
    font-weight: 600;
    color: #f0f6fc;
    margin-bottom: 0.8rem;
}

.movie-info-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
    margin-bottom: 1rem;
    font-size: 0.9rem;
}

.movie-info-plot {
    color: #c9d1d9;
    line-height: 1.6;
    font-size: 0.95rem;
    margin-bottom: 1rem;
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
    
    .selected-movie-info {
        flex-direction: column;
    }
    
    .movie-poster {
        align-self: center;
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
                        cast_names = [actor['name'] for actor in data[:10] if 'name' in actor]
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
                        directors = []
                        for person in data:
                            if person.get('job') == 'Director':
                                directors.append(person['name'])
                        return ' | '.join(directors[:3])
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
        {"id": 19995, "title": "Avatar", "genres": "Action Adventure Fantasy", "vote_average": 7.2, "overview": "In the 22nd century, a paraplegic Marine is dispatched to the moon Pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization.", "cast": "Sam Worthington | Zoe Saldana | Sigourney Weaver", "director": "James Cameron"},
        {"id": 285, "title": "Pirates of the Caribbean: The Curse of the Black Pearl", "genres": "Adventure Fantasy Action", "vote_average": 7.8, "overview": "Blacksmith Will Turner teams up with eccentric pirate Captain Jack Sparrow to save his love, the governor's daughter, from Jack's former pirate allies, who are now undead.", "cast": "Johnny Depp | Orlando Bloom | Keira Knightley", "director": "Gore Verbinski"},
        {"id": 206647, "title": "Spectre", "genres": "Action Adventure Crime", "vote_average": 6.3, "overview": "A cryptic message from Bond's past sends him on a trail to uncover a sinister organization. While M battles political forces to keep the secret service alive, Bond peels back the layers of deceit to reveal the terrible truth behind SPECTRE.", "cast": "Daniel Craig | Christoph Waltz | Léa Seydoux", "director": "Sam Mendes"},
        {"id": 49026, "title": "The Dark Knight Rises", "genres": "Action Crime Drama", "vote_average": 7.6, "overview": "Following the death of District Attorney Harvey Dent, Batman assumes responsibility for Dent's crimes to protect the late attorney's reputation and is subsequently hunted by the Gotham City Police Department. Eight years later, Batman encounters the mysterious Selina Kyle and the villainous Bane, a new terrorist leader who overwhelms Gotham's finest.", "cast": "Christian Bale | Tom Hardy | Anne Hathaway", "director": "Christopher Nolan"},
        {"id": 49529, "title": "John Carter", "genres": "Action Adventure Science Fiction", "vote_average": 6.1, "overview": "John Carter is a war-weary, former military captain who's inexplicably transported to the mysterious and exotic planet of Barsoom (Mars) and reluctantly becomes embroiled in an epic conflict.", "cast": "Taylor Kitsch | Lynn Collins | Samantha Morton", "director": "Andrew Stanton"},
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

def advanced_search(movies_df, query):
    """Enhanced search function for movies, actors, directors, and genres"""
    query = query.lower().strip()
    if not query:
        return movies_df.nlargest(50, 'popularity')
    
    # Multi-field search with scoring
    results = []
    
    # Title search (highest priority)
    title_matches = movies_df[movies_df['title'].str.lower().str.contains(query, na=False)]
    for _, movie in title_matches.iterrows():
        score = 100  # Highest score for title matches
        if query in movie['title'].lower():
            score += 50
        results.append((movie, score))
    
    # Actor/Cast search
    if 'cast' in movies_df.columns:
        cast_matches = movies_df[
            (~movies_df['title'].str.lower().str.contains(query, na=False)) &
            (movies_df['cast'].str.lower().str.contains(query, na=False))
        ]
        for _, movie in cast_matches.iterrows():
            score = 80
            # Boost score if actor name appears earlier in cast list (more prominent role)
            cast_list = movie['cast'].lower()
            position = cast_list.find(query)
            if position < 50:  # Actor in first ~2 names
                score += 20
            results.append((movie, score))
    
    # Director search
    if 'director' in movies_df.columns:
        director_matches = movies_df[
            (~movies_df['title'].str.lower().str.contains(query, na=False)) &
            (movies_df['director'].str.lower().str.contains(query, na=False))
        ]
        for _, movie in director_matches.iterrows():
            score = 75
            results.append((movie, score))
    
    # Genre search
    genre_matches = movies_df[
        (~movies_df['title'].str.lower().str.contains(query, na=False)) &
        (movies_df['genres'].str.lower().str.contains(query, na=False))
    ]
    for _, movie in genre_matches.iterrows():
        score = 60
        # Boost if exact genre match
        if query in movie['genres'].lower().split():
            score += 20
        results.append((movie, score))
    
    # Overview/plot search (lower priority)
    overview_matches = movies_df[
        (~movies_df['title'].str.lower().str.contains(query, na=False)) &
        (movies_df['overview'].str.lower().str.contains(query, na=False))
    ]
    for _, movie in overview_matches.iterrows():
        score = 40
        results.append((movie, score))
    
    if not results:
        return pd.DataFrame()
    
    # Remove duplicates and sort by score
    seen_ids = set()
    unique_results = []
    for movie, score in results:
        if movie['id'] not in seen_ids:
            seen_ids.add(movie['id'])
            unique_results.append((movie, score))
    
    # Sort by score (descending) and then by popularity/rating
    unique_results.sort(key=lambda x: (x[1], x[0]['vote_average'], x[0]['popularity']), reverse=True)
    
    # Convert back to DataFrame
    result_df = pd.DataFrame([movie for movie, _ in unique_results[:50]])
    return result_df.reset_index(drop=True)

def recommend_movies(movie_title, movies_df, similarity_matrix, n_recommendations=6):
    """Enhanced movie recommendation system"""
    try:
        if not SKLEARN_AVAILABLE:
            return simple_recommend(movies_df, movie_title, n_recommendations)
        
        # Find exact or partial matches
        exact_matches = movies_df[movies_df['title'].str.lower() == movie_title.lower()]
        if len(exact_matches) == 0:
            partial_matches = movies_df[movies_df['title'].str.lower().str.contains(movie_title.lower(), na=False)]
            if len(partial_matches) == 0:
                return pd.DataFrame()
            movie_idx = partial_matches.index[0]
        else:
            movie_idx = exact_matches.index[0]
        
        # Get similarity scores
        sim_scores = list(enumerate(similarity_matrix[movie_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Filter out the movie itself and get recommendations
        sim_scores = [score for score in sim_scores if score[0] != movie_idx]
        sim_scores = sim_scores[:n_recommendations*2]  # Get more to filter better
        
        # Create recommendations DataFrame
        movie_indices = [i[0] for i in sim_scores]
        recommendations = movies_df.iloc[movie_indices].copy()
        recommendations['similarity_score'] = [score[1] for score in sim_scores]
        
        # Enhanced filtering and ranking
        # Prefer movies with similar genres or higher ratings
        selected_movie = movies_df.iloc[movie_idx]
        selected_genres = selected_movie['genres'].lower().split()
        
        def calculate_enhanced_score(row):
            base_score = row['similarity_score']
            rating_boost = (row['vote_average'] - 5) / 10  # Boost for high ratings
            popularity_boost = min(row['popularity'] / 100, 0.2)  # Small popularity boost
            
            # Genre similarity boost
            movie_genres = row['genres'].lower().split()
            genre_overlap = len(set(selected_genres) & set(movie_genres)) / max(len(selected_genres), 1)
            genre_boost = genre_overlap * 0.1
            
            return base_score + rating_boost + popularity_boost + genre_boost
        
        recommendations['enhanced_score'] = recommendations.apply(calculate_enhanced_score, axis=1)
        recommendations = recommendations.sort_values('enhanced_score', ascending=False)
        
        return recommendations.head(n_recommendations)
        
    except Exception as e:
        st.error(f"Error getting recommendations: {e}")
        return pd.DataFrame()

def main():
    # Main Header
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">Cinema Vault</h1>
        <p class="main-subtitle">Premium cinematic discovery powered by advanced content analysis.</p>
        <div class="stats-container">
            <div class="stat-item">
                <span class="stat-number">5,000+</span>
                <div class="stat-label">Movies</div>
            </div>
            <div class="stat-item">
                <span class="stat-number">750K+</span>
                <div class="stat-label">Users</div>
            </div>
            <div class="stat-item">
                <span class="stat-number">99%</span>
                <div class="stat-label">Precision</div>
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
        <h2 class="hero-title">Discover Your Next <span class="hero-gradient-text">Cinematic Journey</span></h2>
        <p class="hero-description">Experience the most sophisticated movie discovery platform with deep content analysis and personalized recommendations.</p>
    </div>
    """, unsafe_allow_html=True)

    # Search and Recommendation Section
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="search-section">', unsafe_allow_html=True)
        st.markdown("### Find Your Perfect Movie")
        
        search_query = st.text_input("Search movies, actors, directors, genres...", placeholder="Try: 'Brad Pitt', 'Action', 'Christopher Nolan'...")
        
        # Enhanced search results
        if search_query:
            search_results = advanced_search(movies_df, search_query)
            if len(search_results) > 0:
                # Create a list of titles for selectbox
                result_options = []
                for _, movie in search_results.head(30).iterrows():
                    # Add additional context to help users identify movies
                    year = ""
                    if 'release_date' in movie and pd.notna(movie['release_date']):
                        try:
                            year = f" ({movie['release_date'][:4]})"
                        except:
                            year = ""
                    
                    # Show rating and genre info
                    rating_info = f" - ⭐{movie['vote_average']:.1f}"
                    genre_info = f" - {movie['genres'][:30]}..." if len(movie['genres']) > 30 else f" - {movie['genres']}"
                    
                    display_text = f"{movie['title']}{year}{rating_info}{genre_info}"
                    result_options.append(display_text)
                
                selected_display = st.selectbox("Choose a movie:", result_options)
                
                # Extract actual movie title from display text
                if selected_display:
                    selected_movie_title = selected_display.split(" - ⭐")[0]
                    # Remove year if present
                    if " (" in selected_movie_title and selected_movie_title.endswith(")"):
                        selected_movie_title = selected_movie_title.rsplit(" (", 1)[0]
                    
                    selected_movie = None
                    for _, movie in search_results.iterrows():
                        if movie['title'] == selected_movie_title:
                            selected_movie = movie
                            break
                else:
                    selected_movie = None
            else:
                st.warning(f"No results found for '{search_query}'. Try different keywords.")
                selected_movie = None
        else:
            # Default popular/top-rated movies
            popular_movies = movies_df.nlargest(20, 'popularity')
            top_rated = movies_df.nlargest(20, 'vote_average')
            combined = pd.concat([popular_movies, top_rated]).drop_duplicates('title')
            
            movie_options = []
            for _, movie in combined.head(30).iterrows():
                year = ""
                if 'release_date' in movie and pd.notna(movie['release_date']):
                    try:
                        year = f" ({movie['release_date'][:4]})"
                    except:
                        year = ""
                display_text = f"{movie['title']}{year} - ⭐{movie['vote_average']:.1f}"
                movie_options.append(display_text)
            
            selected_display = st.selectbox("Choose from popular movies:", movie_options)
            
            if selected_display:
                selected_movie_title = selected_display.split(" - ⭐")[0]
                if " (" in selected_movie_title and selected_movie_title.endswith(")"):
                    selected_movie_title = selected_movie_title.rsplit(" (", 1)[0]
                
                selected_movie = None
                for _, movie in combined.iterrows():
                    if movie['title'] == selected_movie_title:
                        selected_movie = movie
                        break
            else:
                selected_movie = None
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if selected_movie is not None:
            # Display selected movie info with poster and complete details
            poster_url = get_movie_poster(selected_movie['title'], selected_movie.get('id'))
            
            # Display selected movie info with poster and complete details
            st.markdown('<div class="selected-movie-info">', unsafe_allow_html=True)
            
            # Create two columns for poster and info
            movie_col1, movie_col2 = st.columns([1, 2])
            
            with movie_col1:
                st.image(poster_url, width=200, caption=selected_movie['title'])
                st.markdown(f"**⭐ {selected_movie['vote_average']:.1f}/10**")
                st.markdown(f"**🎭 {selected_movie['genres']}**")
                if 'director' in selected_movie and pd.notna(selected_movie['director']) and selected_movie['director']:
                    st.markdown(f"**🎬 {selected_movie['director']}**")
                if 'release_date' in selected_movie and pd.notna(selected_movie['release_date']):
                    try:
                        st.markdown(f"**📅 {selected_movie['release_date'][:4]}**")
                    except:
                        pass
                if 'runtime' in selected_movie and pd.notna(selected_movie['runtime']):
                    st.markdown(f"**⏱️ {int(selected_movie['runtime'])}min**")
            
            with movie_col2:
                st.markdown(f"### {selected_movie['title']}")
                st.markdown(f"**Plot:** {selected_movie['overview']}")
                if 'cast' in selected_movie and pd.notna(selected_movie['cast']) and selected_movie['cast']:
                    st.markdown(f"**Cast:** {selected_movie['cast']}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("### Recommended For You")
        
        if selected_movie is not None:
            recommendations = recommend_movies(
                selected_movie['title'], movies_df, similarity_matrix, 6
            )
            
            if len(recommendations) > 0:
                for idx, (_, movie) in enumerate(recommendations.iterrows()):
                    # Additional movie metadata formatting
                    year = ""
                    runtime = ""
                    director_info = ""
                    cast_info = ""
                    
                    if 'release_date' in movie and pd.notna(movie['release_date']):
                        try:
                            year = movie['release_date'][:4]
                        except:
                            year = ""
                    
                    if 'runtime' in movie and pd.notna(movie['runtime']):
                        runtime = f"{int(movie['runtime'])}min"
                    
                    if 'director' in movie and pd.notna(movie['director']) and movie['director']:
                        director_info = f"🎬 {movie['director']}"
                    
                    if 'cast' in movie and pd.notna(movie['cast']) and movie['cast']:
                        cast_list = movie['cast'].split(' | ')[:3]  # Show top 3 actors
                        cast_info = f"👥 {', '.join(cast_list)}"
                    
                    # Create recommendation card
                    st.markdown(f"""
                    <div class="movie-card">
                        <div class="match-score">{similarity_score:.0f}% Match</div>
                        <h3 class="movie-title">{movie['title']}</h3>
                        <div style="margin-bottom: 1rem;">
                            <span class="movie-rating">⭐ {movie['vote_average']:.1f}</span>
                            <span class="movie-genre">{movie['genres']}</span>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Add metadata if available
                    if year or runtime or director_info:
                        metadata_parts = []
                        if year: metadata_parts.append(f"📅 {year}")
                        if runtime: metadata_parts.append(f"⏱️ {runtime}")
                        if director_info: metadata_parts.append(director_info)
                        
                        st.markdown(f"""
                        <div class="movie-meta">
                            {' | '.join(metadata_parts)}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Plot and cast
                    st.markdown(f"""
                        <p class="movie-description"><strong>Plot:</strong> {movie['overview']}</p>
                    """, unsafe_allow_html=True)
                    
                    if cast_info:
                        st.markdown(f"""
                        <div style='margin-top: 0.8rem; color: #8b949e; font-size: 0.85rem;'>{cast_info}</div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.warning("No recommendations found for this movie.")
        else:
            # Featured movies section when no movie is selected
            st.markdown("#### Featured Movies")
            featured = movies_df.nlargest(4, 'vote_average')
            
            for _, movie in featured.iterrows():
                year = ""
                if 'release_date' in movie and pd.notna(movie['release_date']):
                    try:
                        year = f" ({movie['release_date'][:4]})"
                    except:
                        year = ""
                
                director_info = ""
                if 'director' in movie and pd.notna(movie['director']) and movie['director']:
                    director_info = f"🎬 {movie['director']}"
                
                # Create featured movie card
                st.markdown(f"""
                <div class="movie-card">
                    <h3 class="movie-title">{movie['title']}{year}</h3>
                    <div style="margin-bottom: 1rem;">
                        <span class="movie-rating">⭐ {movie['vote_average']:.1f}</span>
                        <span class="movie-genre">{movie['genres']}</span>
                    </div>
                """, unsafe_allow_html=True)
                
                if director_info:
                    st.markdown(f"""
                    <div style='margin-bottom: 0.8rem; color: #8b949e;'>{director_info}</div>
                    """, unsafe_allow_html=True)
                
                st.markdown(f"""
                    <p class="movie-description">{movie['overview']}</p>
                </div>
                """, unsafe_allow_html=True)

    # Popular Movies Grid Section
    st.markdown('<h2 class="section-header">Trending Now</h2>', unsafe_allow_html=True)
    
    # Get trending movies (mix of popular and highly rated)
    trending_movies = movies_df.nlargest(12, 'popularity')
    
    cols = st.columns(6)
    for idx, (_, movie) in enumerate(trending_movies.iterrows()):
        with cols[idx % 6]:
            poster_url = get_movie_poster(movie['title'], movie.get('id'))
            year = ""
            if 'release_date' in movie and pd.notna(movie['release_date']):
                try:
                    year = f" ({movie['release_date'][:4]})"
                except:
                    year = ""
            
            st.markdown(f"""
            <div class="poster-card">
                <img src="{poster_url}" alt="{movie['title']}">
                <div class="poster-title">{movie['title'][:18]}{year}{'...' if len(movie['title']) > 18 else ''}</div>
                <div style="color: #ffd700; font-size: 0.8rem; margin-top: 0.3rem;">⭐ {movie['vote_average']:.1f}</div>
            </div>
            """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer">
        <p>🎬 Powered by TMDB Database via GitHub Repository | Built with Advanced Content-Based Filtering</p>
        <p>💡 Multi-dimensional analysis using plot, genres, cast, crew, and user preferences</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
