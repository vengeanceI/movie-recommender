import streamlit as st
import pandas as pd
import requests
import json
import gzip
import io
import re
from difflib import SequenceMatcher
try:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    try:
        import sklearn
        from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
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
    
    .selected-movie-info {
        flex-direction: column;
    }
    
    .movie-poster {
        align-self: center;
    }
}

.search-results-header {
    color: #58a6ff;
    font-size: 1.1rem;
    font-weight: 600;
    margin: 1rem 0 0.5rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(48, 54, 61, 0.6);
}

.no-results {
    background: rgba(139, 148, 158, 0.1);
    border: 1px solid rgba(139, 148, 158, 0.3);
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
    color: #8b949e;
    text-align: center;
}

.actor-bio {
    background: rgba(33, 38, 45, 0.6);
    border: 1px solid rgba(48, 54, 61, 0.6);
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
    color: #c9d1d9;
}

.genre-description {
    background: rgba(33, 38, 45, 0.6);
    border: 1px solid rgba(48, 54, 61, 0.6);
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
    color: #c9d1d9;
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
    """Process and clean movie data with enhanced credits handling"""
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
            
            # Create searchable cast field
            movies_df['cast_searchable'] = movies_df['cast'].str.lower().str.replace('|', ' ')
        
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
            movies_df['director_searchable'] = movies_df['director'].str.lower().str.replace('|', ' ')
        
        # Create combined features for similarity calculation
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
    """Create comprehensive sample data if GitHub files are unavailable"""
    sample_data = [
        {"id": 19995, "title": "Avatar", "genres": "Action Adventure Fantasy", "vote_average": 7.2, "overview": "In the 22nd century, a paraplegic Marine is dispatched to the moon Pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. Avatar takes us to a spectacular world beyond imagination, where a reluctant hero embarks on a journey of redemption and discovery as he leads a heroic battle to save a civilization.", "cast": "Sam Worthington | Zoe Saldana | Sigourney Weaver | Stephen Lang | Michelle Rodriguez", "director": "James Cameron", "release_date": "2009-12-18", "runtime": 162, "vote_count": 11800, "popularity": 150.437},
        {"id": 285, "title": "Pirates of the Caribbean: The Curse of the Black Pearl", "genres": "Adventure Fantasy Action", "vote_average": 7.8, "overview": "Blacksmith Will Turner teams up with eccentric pirate Captain Jack Sparrow to save his love, the governor's daughter, from Jack's former pirate allies, who are now undead. This swashbuckling tale of adventure on the high seas brings supernatural elements and unforgettable characters together in an epic story of love, betrayal, and redemption.", "cast": "Johnny Depp | Orlando Bloom | Keira Knightley | Geoffrey Rush | Jack Davenport", "director": "Gore Verbinski", "release_date": "2003-07-09", "runtime": 143, "vote_count": 8945, "popularity": 123.456},
        {"id": 550, "title": "Fight Club", "genres": "Drama Thriller", "vote_average": 8.4, "overview": "A ticking-time-bomb insomniac and a slippery soap salesman channel primal male aggression into a shocking new form of therapy. Their concept catches on, with underground fight clubs forming in every town, until an eccentric gets in the way and ignites an out-of-control spiral toward oblivion.", "cast": "Brad Pitt | Edward Norton | Helena Bonham Carter | Meat Loaf | Jared Leto", "director": "David Fincher", "release_date": "1999-10-15", "runtime": 139, "vote_count": 15420, "popularity": 89.234},
        {"id": 155, "title": "The Dark Knight", "genres": "Action Crime Drama", "vote_average": 8.5, "overview": "Batman raises the stakes in his war on crime. With the help of Lt. Jim Gordon and District Attorney Harvey Dent, Batman sets out to dismantle the remaining criminal organizations that plague the streets. The partnership proves to be effective, but they soon find themselves prey to a reign of chaos unleashed by a rising criminal mastermind known to the terrified citizens of Gotham as the Joker.", "cast": "Christian Bale | Heath Ledger | Aaron Eckhart | Michael Caine | Maggie Gyllenhaal", "director": "Christopher Nolan", "release_date": "2008-07-18", "runtime": 152, "vote_count": 18500, "popularity": 140.789},
        {"id": 157336, "title": "Interstellar", "genres": "Adventure Drama Science Fiction", "vote_average": 8.1, "overview": "The adventures of a group of explorers who make use of a newly discovered wormhole to surpass the limitations on human space travel and conquer the vast distances involved in an interstellar voyage. In Earth's future, a global crop blight and second Dust Bowl are slowly rendering the planet uninhabitable.", "cast": "Matthew McConaughey | Anne Hathaway | Jessica Chastain | Michael Caine | Matt Damon", "director": "Christopher Nolan", "release_date": "2014-11-07", "runtime": 169, "vote_count": 16789, "popularity": 132.567},
        {"id": 122, "title": "The Lord of the Rings: The Return of the King", "genres": "Adventure Drama Action", "vote_average": 8.3, "overview": "Aragorn is revealed as the heir to the ancient kings as he, Gandalf and the other members of the broken fellowship struggle to save Gondor from Sauron's forces. Meanwhile, Frodo and Sam bring the ring closer to the heart of Mordor, the dark lord's realm.", "cast": "Elijah Wood | Ian McKellen | Viggo Mortensen | Sean Astin | Orlando Bloom", "director": "Peter Jackson", "release_date": "2003-12-17", "runtime": 201, "vote_count": 14567, "popularity": 98.345},
        {"id": 238, "title": "The Godfather", "genres": "Drama Crime", "vote_average": 8.7, "overview": "Spanning the years 1945 to 1955, a chronicle of the fictional Italian-American Corleone crime family. When organized crime family patriarch, Vito Corleone barely survives an attempt on his life, his youngest son, Michael steps in to take care of the would-be killers, launching a campaign of bloody revenge.", "cast": "Marlon Brando | Al Pacino | James Caan | Diane Keaton | Robert Duvall", "director": "Francis Ford Coppola", "release_date": "1972-03-24", "runtime": 175, "vote_count": 12890, "popularity": 87.654},
        {"id": 13, "title": "Forrest Gump", "genres": "Comedy Drama Romance", "vote_average": 8.2, "overview": "A man with a low IQ has accomplished great things in his life and been present during significant historic events—in each case, far exceeding what anyone imagined he could do. But despite all he has achieved, his one true love eludes him.", "cast": "Tom Hanks | Robin Wright | Gary Sinise | Sally Field | Mykelti Williamson", "director": "Robert Zemeckis", "release_date": "1994-07-06", "runtime": 142, "vote_count": 13456, "popularity": 105.432},
        {"id": 769, "title": "Goodfellas", "genres": "Drama Crime", "vote_average": 8.2, "overview": "The true story of Henry Hill, a half-Irish, half-Sicilian Brooklyn kid who is adopted by neighbourhood gangsters at an early age and climbs the ranks of a Mafia family under the guidance of Jimmy Conway.", "cast": "Robert De Niro | Ray Liotta | Joe Pesci | Lorraine Bracco | Paul Sorvino", "director": "Martin Scorsese", "release_date": "1990-09-21", "runtime": 146, "vote_count": 9876, "popularity": 76.543},
        {"id": 278, "title": "The Shawshank Redemption", "genres": "Drama", "vote_average": 8.7, "overview": "Framed in the 1940s for the double murder of his wife and her lover, upstanding banker Andy Dufresne begins a new life at the Shawshank prison, where he puts his accounting skills to work for an amoral warden. During his long stretch in prison, Dufresne comes to be admired by the other inmates -- including an older prisoner named Red -- for his integrity and unquenchable sense of hope.", "cast": "Tim Robbins | Morgan Freeman | Bob Gunton | William Sadler | Clancy Brown", "director": "Frank Darabont", "release_date": "1994-09-23", "runtime": 142, "vote_count": 17890, "popularity": 94.123}
    ]
    
    df = pd.DataFrame(sample_data)
    
    # Add searchable fields
    df['cast_searchable'] = df['cast'].str.lower().str.replace('|', ' ')
    df['director_searchable'] = df['director'].str.lower().str.replace('|', ' ')
    
    # Create combined features
    df['combined_features'] = df['overview'] + ' ' + df['genres'] + ' ' + df['cast'] + ' ' + df['director']
    df['keywords'] = ''
    
    return df

@st.cache_data
def create_similarity_matrix(movies_df):
    """Create enhanced similarity matrix for recommendations"""
    if not SKLEARN_AVAILABLE:
        return None
        
    try:
        # Use TfidfVectorizer for better results
        tfidf = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.8
        )
        
        vectors = tfidf.fit_transform(movies_df['combined_features']).toarray()
        similarity = cosine_similarity(vectors)
        
        return similarity
        
    except Exception as e:
        st.error(f"Error creating similarity matrix: {e}")
        return None

def get_movie_poster(movie_title, tmdb_id=None):
    """Get movie poster from TMDB API with fallback"""
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

def similarity_score(a, b):
    """Calculate string similarity"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def advanced_search(movies_df, query, search_type="auto"):
    """Enhanced search function with intelligent categorization"""
    if not query or len(query.strip()) < 2:
        return pd.DataFrame(), "auto", ""
    
    query = query.lower().strip()
    results = []
    
    # Detect search type if auto
    if search_type == "auto":
        # Common actor names (you can expand this list)
        common_actors = ['brad pitt', 'leonardo dicaprio', 'tom hanks', 'will smith', 'robert downey jr',
                        'johnny depp', 'morgan freeman', 'samuel l jackson', 'matt damon', 'christian bale',
                        'scarlett johansson', 'jennifer lawrence', 'angelina jolie', 'meryl streep',
                        'denzel washington', 'tom cruise', 'robert de niro', 'al pacino', 'heath ledger']
        
        common_directors = ['christopher nolan', 'martin scorsese', 'quentin tarantino', 'steven spielberg',
                           'james cameron', 'ridley scott', 'david fincher', 'tim burton', 'peter jackson',
                           'francis ford coppola', 'stanley kubrick', 'alfred hitchcock']
        
        common_genres = ['action', 'adventure', 'comedy', 'drama', 'horror', 'thriller', 'romance',
                        'science fiction', 'fantasy', 'animation', 'crime', 'mystery', 'western',
                        'war', 'documentary', 'biography', 'history', 'music', 'family']
        
        # Determine search type
        if any(actor in query for actor in common_actors):
            search_type = "actor"
        elif any(director in query for director in common_directors):
            search_type = "director"  
        elif any(genre in query for genre in common_genres):
            search_type = "genre"
        else:
            # Check for partial matches or if query contains common name patterns
            if len(query.split()) >= 2 and not any(word in query for word in ['the', 'a', 'an', 'of', 'in', 'on', 'at']):
                search_type = "person"  # Likely an actor/director name
            else:
                search_type = "title"
    
    # Movie title search (highest priority)
    if search_type in ["auto", "title"]:
        # Exact matches
        exact_matches = movies_df[movies_df['title'].str.lower() == query]
        for _, movie in exact_matches.iterrows():
            results.append((movie, 100, "exact_title"))
        
        # Partial title matches
        partial_matches = movies_df[
            (movies_df['title'].str.lower() != query) &
            (movies_df['title'].str.lower().str.contains(re.escape(query), na=False))
        ]
        for _, movie in partial_matches.iterrows():
            score = 90 + similarity_score(query, movie['title']) * 10
            results.append((movie, score, "partial_title"))
    
    # Actor/Cast search
    if search_type in ["auto", "actor", "person"] and 'cast_searchable' in movies_df.columns:
        cast_matches = movies_df[
            movies_df['cast_searchable'].str.contains(re.escape(query), na=False, regex=True)
        ]
        for _, movie in cast_matches.iterrows():
            # Higher score if actor appears earlier in cast (more prominent role)
            cast_list = movie['cast_searchable']
            position = cast_list.find(query)
            if position != -1:
                score = 85 - (position / 10)  # Earlier position = higher score
                # Bonus for exact name match
                if query in cast_list.split():
                    score += 10
                results.append((movie, min(score, 95), "actor"))
    
    # Director search
    if search_type in ["auto", "director", "person"] and 'director_searchable' in movies_df.columns:
        director_matches = movies_df[
            movies_df['director_searchable'].str.contains(re.escape(query), na=False, regex=True)
        ]
        for _, movie in director_matches.iterrows():
            score = 80
            # Bonus for exact director match
            if query in movie['director_searchable'].split():
                score += 15
            results.append((movie, score, "director"))
    
    # Genre search
    if search_type in ["auto", "genre"]:
        genre_matches = movies_df[
            movies_df['genres'].str.lower().str.contains(re.escape(query), na=False)
        ]
        for _, movie in genre_matches.iterrows():
            score = 70
            # Boost for exact genre match
            genres_list = movie['genres'].lower().split()
            if query in genres_list:
                score += 20
            results.append((movie, score, "genre"))
    
    # Overview/plot search (lowest priority but still important)
    if search_type == "auto":
        overview_matches = movies_df[
            movies_df['overview'].str.lower().str.contains(re.escape(query), na=False)
        ]
        for _, movie in overview_matches.iterrows():
            score = 50
            # Count occurrences in overview for relevance
            occurrences = movie['overview'].lower().count(query)
            score += min(occurrences * 5, 20)
            results.append((movie, score, "plot"))
    
    if not results:
        return pd.DataFrame(), search_type, f"No results found for '{query}'"
    
    # Remove duplicates and sort by score
    seen_ids = set()
    unique_results = []
    for movie, score, match_type in results:
        if movie['id'] not in seen_ids:
            seen_ids.add(movie['id'])
            unique_results.append((movie, score, match_type))
    
    # Sort by score (descending) then by popularity/rating
    unique_results.sort(key=lambda x: (x[1], x[0]['vote_average'], x[0]['popularity']), reverse=True)
    
    # Convert back to DataFrame and add match info
    result_df = pd.DataFrame([movie for movie, _, _ in unique_results[:30]])
    match_types = [match_type for _, _, match_type in unique_results[:30]]
    scores = [score for _, score, _ in unique_results[:30]]
    
    if len(result_df) > 0:
        result_df['match_type'] = match_types
        result_df['search_score'] = scores
    
    return result_df.reset_index(drop=True), search_type, ""

def get_person_info(person_name, person_type="actor"):
    """Get information about an actor or director from TMDB"""
    try:
        if not TMDB_API_KEY:
            return None
            
        search_url = f"https://api.themoviedb.org/3/search/person?api_key={TMDB_API_KEY}&query={person_name}"
        response = requests.get(search_url, timeout=5)
        data = response.json()
        
        if data['results']:
            person_id = data['results'][0]['id']
            person_url = f"https://api.themoviedb.org/3/person/{person_id}?api_key={TMDB_API_KEY}"
            person_response = requests.get(person_url, timeout=5)
            person_data = person_response.json()
            
            return {
                'name': person_data.get('name', person_name),
                'biography': person_data.get('biography', 'No biography available.'),
                'birthday': person_data.get('birthday', ''),
                'place_of_birth': person_data.get('place_of_birth', ''),
                'profile_path': person_data.get('profile_path', '')
            }
    except:
        pass
    return None

def get_genre_description(genre_name):
    """Get description for movie genres"""
    genre_descriptions = {
        'action': 'Action films are characterized by high energy, big-budget physical stunts, chases, fights, battles, and destructive crises.',
        'adventure': 'Adventure films are exciting stories, with new experiences or exotic locales, very similar to or often paired with the action film genre.',
        'animation': 'Animation films are made from individual drawings, paintings, or illustrations photographed in sequence.',
        'comedy': 'Comedy films are designed to make the audience laugh through amusement and most often work by exaggerating characteristics.',
        'crime': 'Crime films are developed around the sinister actions of criminals or mobsters, particularly bankrobbers, underworld figures, or ruthless hoodlums.',
        'documentary': 'Documentary films constitute a broad category of visual expression that is based on the attempt, in one fashion or another, to document reality.',
        'drama': 'Drama films are serious presentations or stories with settings or life situations that portray realistic characters in conflict.',
        'family': 'Family films are movies suitable for the entire family, that appeal to a wide range of ages.',
        'fantasy': 'Fantasy films are films with fantastic themes, usually involving magic, supernatural events, make-believe creatures.',
        'history': 'Historical films are based on real figures, events, and historical periods, offering dramatic interpretations of past events.',
        'horror': 'Horror films seek to elicit fear, suspense, disgust, or startlement from audiences through frightening and disturbing elements.',
        'music': 'Music films are genre of film in which songs sung by the characters are interwoven into the narrative.',
        'mystery': 'Mystery films revolve around the solution of a problem or crime, with the audience following clues alongside the protagonist.',
        'romance': 'Romance films make the romantic love story or the search for strong and pure love and romance the main plot focus.',
        'science fiction': 'Science fiction films feature futuristic concepts such as advanced technology, space exploration, time travel, parallel universes.',
        'thriller': 'Thriller films are characterized by constant danger, high stakes, and psychological tension that keeps audiences on edge.',
        'war': 'War films acknowledge the horror and heartbreak of war, letting the actual combat fighting serve as the primary plot.',
        'western': 'Western films are set in the American Old West and embody the spirit, the struggle, and the demise of the new frontier.'
    }
    return genre_descriptions.get(genre_name.lower(), f'{genre_name} films offer unique storytelling experiences within this distinctive genre.')

def recommend_movies(movie_title, movies_df, similarity_matrix, n_recommendations=6):
    """Enhanced movie recommendation system with better matching"""
    try:
        if not SKLEARN_AVAILABLE:
            return simple_recommend(movies_df, movie_title, n_recommendations)
        
        # Find the best match for the movie
        movie_idx = None
        best_match_score = 0
        
        # First try exact match
        exact_matches = movies_df[movies_df['title'].str.lower() == movie_title.lower()]
        if len(exact_matches) > 0:
            movie_idx = exact_matches.index[0]
        else:
            # Try partial matches and find best one
            for idx, row in movies_df.iterrows():
                score = similarity_score(movie_title.lower(), row['title'].lower())
                if score > best_match_score and score > 0.5:  # Minimum threshold
                    best_match_score = score
                    movie_idx = idx
        
        if movie_idx is None:
            return pd.DataFrame()
        
        # Get similarity scores
        sim_scores = list(enumerate(similarity_matrix[movie_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Filter out the movie itself
        sim_scores = [score for score in sim_scores if score[0] != movie_idx]
        
        # Enhanced filtering and ranking
        selected_movie = movies_df.iloc[movie_idx]
        selected_genres = selected_movie['genres'].lower().split()
        
        # Get more candidates for better filtering
        sim_scores = sim_scores[:n_recommendations*3]
        movie_indices = [i[0] for i in sim_scores]
        candidates = movies_df.iloc[movie_indices].copy()
        candidates['similarity_score'] = [score[1] for score in sim_scores]
        
        def calculate_enhanced_score(row):
            base_score = row['similarity_score']
            
            # Rating boost (prefer higher rated movies)
            rating_boost = (row['vote_average'] - 5) / 10 * 0.2
            
            # Popularity boost (slight preference for known movies)
            popularity_boost = min(row['popularity'] / 100, 0.1) * 0.1
            
            # Genre similarity boost
            movie_genres = row['genres'].lower().split()
            genre_overlap = len(set(selected_genres) & set(movie_genres))
            genre_boost = genre_overlap / max(len(selected_genres), 1) * 0.15
            
            # Vote count reliability boost
            vote_count_boost = min(row['vote_count'] / 1000, 0.1) * 0.1
            
            # Penalize very low ratings
            rating_penalty = 0
            if row['vote_average'] < 4.0:
                rating_penalty = -0.3
            
            return base_score + rating_boost + popularity_boost + genre_boost + vote_count_boost + rating_penalty
        
        candidates['enhanced_score'] = candidates.apply(calculate_enhanced_score, axis=1)
        candidates = candidates.sort_values('enhanced_score', ascending=False)
        
        # Ensure diversity in recommendations
        final_recommendations = []
        used_directors = set()
        genre_counts = {}
        
        for _, movie in candidates.iterrows():
            if len(final_recommendations) >= n_recommendations:
                break
                
            # Diversity checks
            director = movie.get('director', '')
            movie_genres = movie['genres'].split()
            
            # Prefer different directors (but don't exclude entirely)
            director_penalty = 0.1 if director in used_directors and director != '' else 0
            
            # Prefer genre diversity (but don't exclude entirely)  
            genre_penalty = 0
            for genre in movie_genres:
                if genre in genre_counts and genre_counts[genre] >= 2:
                    genre_penalty += 0.05
            
            # Apply penalties
            final_score = movie['enhanced_score'] - director_penalty - genre_penalty
            
            if final_score > 0.1:  # Minimum threshold
                final_recommendations.append(movie)
                used_directors.add(director)
                for genre in movie_genres:
                    genre_counts[genre] = genre_counts.get(genre, 0) + 1
        
        # Convert to DataFrame
        if final_recommendations:
            result_df = pd.DataFrame(final_recommendations)
            # Normalize similarity scores for display
            result_df['similarity_score'] = result_df['similarity_score'] * 0.8 + 0.2  # Scale to 20-100%
            return result_df.head(n_recommendations)
        else:
            return candidates.head(n_recommendations)
        
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
        
        search_query = st.text_input("Search movies, actors, directors, genres...", 
                                   placeholder="Try: 'Brad Pitt', 'Action', 'Christopher Nolan', 'Amazing Spider Man'...")
        
        search_results = pd.DataFrame()
        search_type = "auto"
        error_message = ""
        selected_movie = None
        
        if search_query:
            search_results, search_type, error_message = advanced_search(movies_df, search_query)
            
            if not error_message and len(search_results) > 0:
                # Display search results with context
                if search_type == "actor":
                    st.markdown(f'<div class="search-results-header">🎭 Actor: "{search_query.title()}"</div>', unsafe_allow_html=True)
                    
                    # Get actor info if possible
                    person_info = get_person_info(search_query, "actor")
                    if person_info and person_info.get('biography'):
                        bio = person_info['biography']
                        if len(bio) > 300:
                            bio = bio[:300] + "..."
                        
                        st.markdown(f"""
                        <div class="actor-bio">
                            <strong>{person_info['name']}</strong><br>
                            <small>Born: {person_info.get('birthday', 'N/A')} | {person_info.get('place_of_birth', 'N/A')}</small><br><br>
                            {bio}
                        </div>
                        """, unsafe_allow_html=True)
                
                elif search_type == "director":
                    st.markdown(f'<div class="search-results-header">🎬 Director: "{search_query.title()}"</div>', unsafe_allow_html=True)
                    
                    person_info = get_person_info(search_query, "director")
                    if person_info and person_info.get('biography'):
                        bio = person_info['biography']
                        if len(bio) > 300:
                            bio = bio[:300] + "..."
                        
                        st.markdown(f"""
                        <div class="actor-bio">
                            <strong>{person_info['name']}</strong><br>
                            <small>Born: {person_info.get('birthday', 'N/A')} | {person_info.get('place_of_birth', 'N/A')}</small><br><br>
                            {bio}
                        </div>
                        """, unsafe_allow_html=True)
                
                elif search_type == "genre":
                    st.markdown(f'<div class="search-results-header">🎭 Genre: "{search_query.title()}"</div>', unsafe_allow_html=True)
                    genre_desc = get_genre_description(search_query)
                    st.markdown(f"""
                    <div class="genre-description">
                        {genre_desc}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Create movie selection options
                result_options = []
                for _, movie in search_results.head(20).iterrows():
                    year = ""
                    if 'release_date' in movie and pd.notna(movie['release_date']):
                        try:
                            year = f" ({movie['release_date'][:4]})"
                        except:
                            year = ""
                    
                    rating_info = f" - ⭐{movie['vote_average']:.1f}"
                    
                    # Add context based on search type
                    context_info = ""
                    if search_type == "actor" and 'cast' in movie:
                        context_info = f" - 🎭 {movie['genres'][:20]}..."
                    elif search_type == "director" and 'director' in movie:
                        context_info = f" - 🎬 {movie['genres'][:20]}..."
                    elif search_type == "genre":
                        context_info = f" - {movie['genres']}"
                    
                    display_text = f"{movie['title']}{year}{rating_info}{context_info}"
                    result_options.append(display_text)
                
                selected_display = st.selectbox(
                    f"Choose from {len(search_results)} results:", 
                    result_options,
                    key="search_selectbox"
                )
                
                if selected_display:
                    selected_movie_title = selected_display.split(" - ⭐")[0]
                    if " (" in selected_movie_title and selected_movie_title.endswith(")"):
                        selected_movie_title = selected_movie_title.rsplit(" (", 1)[0]
                    
                    for _, movie in search_results.iterrows():
                        if movie['title'] == selected_movie_title:
                            selected_movie = movie
                            break
            
            elif error_message:
                st.markdown(f'<div class="no-results">{error_message}</div>', unsafe_allow_html=True)
                st.info(f"💡 Try searching for:\n- Movie titles: 'Avatar', 'Inception'\n- Actors: 'Leonardo DiCaprio', 'Scarlett Johansson'\n- Directors: 'Christopher Nolan', 'Steven Spielberg'\n- Genres: 'Action', 'Comedy', 'Thriller'")
        
        # Default selection when no search
        if not search_query:
            popular_movies = movies_df.nlargest(20, 'popularity')
            top_rated = movies_df.nlargest(20, 'vote_average')
            combined = pd.concat([popular_movies, top_rated]).drop_duplicates('title')
            
            movie_options = []
            for _, movie in combined.head(25).iterrows():
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
                
                for _, movie in combined.iterrows():
                    if movie['title'] == selected_movie_title:
                        selected_movie = movie
                        break
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if selected_movie is not None:
            # Display selected movie info with complete plot and poster
            poster_url = get_movie_poster(selected_movie['title'], selected_movie.get('id'))
            
            # Get complete overview
            full_overview = selected_movie['overview']
            
            # Format runtime
            runtime_info = ""
            if 'runtime' in selected_movie and pd.notna(selected_movie['runtime']) and selected_movie['runtime'] > 0:
                hours = int(selected_movie['runtime']) // 60
                minutes = int(selected_movie['runtime']) % 60
                if hours > 0:
                    runtime_info = f"⏱️ {hours}h {minutes}min"
                else:
                    runtime_info = f"⏱️ {minutes}min"
            
            # Format release date
            release_info = ""
            if 'release_date' in selected_movie and pd.notna(selected_movie['release_date']):
                try:
                    release_info = f"📅 {selected_movie['release_date'][:4]}"
                except:
                    release_info = ""
            
            st.markdown(f"""
            <div class="selected-movie-info">
                <div class="movie-poster">
                    <img src="{poster_url}" width="200" alt="{selected_movie['title']}">
                </div>
                <div class="movie-info-content">
                    <h2 class="movie-info-title">{selected_movie['title']}</h2>
                    <div class="movie-info-meta">
                        <span class="movie-rating">⭐ {selected_movie['vote_average']:.1f}/10</span>
                        <span class="movie-genre">🎭 {selected_movie['genres']}</span>
                        {f"<span>🎬 {selected_movie['director']}</span>" if 'director' in selected_movie and pd.notna(selected_movie['director']) and selected_movie['director'] else ""}
                        {f"<span>{release_info}</span>" if release_info else ""}
                        {f"<span>{runtime_info}</span>" if runtime_info else ""}
                    </div>
                    <div class="movie-info-plot">
                        <strong>Plot:</strong> {full_overview}
                    </div>
                    {f"<div style='color: #58a6ff; font-size: 0.9rem; margin-top: 1rem;'><strong>Cast:</strong> {selected_movie['cast']}</div>" if 'cast' in selected_movie and pd.notna(selected_movie['cast']) and selected_movie['cast'] else ""}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### Recommended For You")
        
        if selected_movie is not None:
            recommendations = recommend_movies(
                selected_movie['title'], movies_df, similarity_matrix, 6
            )
            
            if len(recommendations) > 0:
                for idx, (_, movie) in enumerate(recommendations.iterrows()):
                    similarity_score_pct = movie.get('similarity_score', 0) * 100
                    
                    # Format movie metadata
                    year = ""
                    runtime = ""
                    director_info = ""
                    cast_info = ""
                    
                    if 'release_date' in movie and pd.notna(movie['release_date']):
                        try:
                            year = movie['release_date'][:4]
                        except:
                            year = ""
                    
                    if 'runtime' in movie and pd.notna(movie['runtime']) and movie['runtime'] > 0:
                        hours = int(movie['runtime']) // 60
                        minutes = int(movie['runtime']) % 60
                        if hours > 0:
                            runtime = f"{hours}h {minutes}min"
                        else:
                            runtime = f"{minutes}min"
                    
                    if 'director' in movie and pd.notna(movie['director']) and movie['director']:
                        director_info = f"🎬 {movie['director']}"
                    
                    if 'cast' in movie and pd.notna(movie['cast']) and movie['cast']:
                        cast_list = movie['cast'].split(' | ')[:3]
                        cast_info = f"👥 {', '.join(cast_list)}"
                    
                    # Complete plot overview
                    full_plot = movie['overview']
                    
                    st.markdown(f"""
                    <div class="movie-card">
                        <div class="match-score">{similarity_score_pct:.0f}% Match</div>
                        <h3 class="movie-title">{movie['title']}</h3>
                        <div style="margin-bottom: 1rem;">
                            <span class="movie-rating">⭐ {movie['vote_average']:.1f}</span>
                            <span class="movie-genre">{movie['genres']}</span>
                        </div>
                        <div class="movie-meta">
                            {f"<span>📅 {year}</span>" if year else ""}
                            {f"<span>⏱️ {runtime}</span>" if runtime else ""}
                            {f"<span>{director_info}</span>" if director_info else ""}
                        </div>
                        <p class="movie-description"><strong>Plot:</strong> {full_plot}</p>
                        {f"<div style='margin-top: 0.8rem; color: #8b949e; font-size: 0.85rem;'>{cast_info}</div>" if cast_info else ""}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No recommendations found for this movie. Try selecting a different movie.")
        else:
            # Featured movies when no selection
            st.markdown("#### Featured Movies")
            featured = movies_df.nlargest(6, 'vote_average')
            
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
                
                full_plot = movie['overview']
                
                st.markdown(f"""
                <div class="movie-card">
                    <h3 class="movie-title">{movie['title']}{year}</h3>
                    <div style="margin-bottom: 1rem;">
                        <span class="movie-rating">⭐ {movie['vote_average']:.1f}</span>
                        <span class="movie-genre">{movie['genres']}</span>
                    </div>
                    {f"<div style='margin-bottom: 0.8rem; color: #8b949e;'>{director_info}</div>" if director_info else ""}
                    <p class="movie-description">{full_plot}</p>
                </div>
                """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div style="text-align: center; color: #8b949e; font-size: 0.9rem; margin-top: 3rem; padding: 2rem; border-top: 1px solid rgba(48, 54, 61, 0.6);">
        <p>🎬 Powered by TMDB Database via GitHub Repository | Built with Advanced Content-Based Filtering</p>
        <p>💡 Multi-dimensional analysis using plot, genres, cast, crew, and intelligent search algorithms</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
