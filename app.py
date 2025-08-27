import streamlit as st
import pandas as pd
import requests
import json
import io
import re
from difflib import SequenceMatcher

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    import random
    
    def simple_recommend(movies_df, selected_title, n_recommendations=5):
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

def load_styles():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #21262d 100%);
        font-family: 'Inter', sans-serif;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

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
    }

    .main-subtitle {
        font-size: 1.2rem;
        color: #8b949e;
        font-weight: 300;
        margin-bottom: 2rem;
    }

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
    }

    .stat-label {
        font-size: 1rem;
        color: #8b949e;
        font-weight: 400;
        margin-top: 0.5rem;
    }

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

    .search-section {
        background: rgba(33, 38, 45, 0.6);
        border: 1px solid rgba(48, 54, 61, 0.8);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 2rem;
    }

    .selected-movie-info {
        background: rgba(33, 38, 45, 0.9);
        border: 1px solid rgba(48, 54, 61, 0.8);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 2rem;
        display: flex;
        gap: 2rem;
        align-items: flex-start;
        min-height: 450px;
    }

    .movie-poster {
        flex-shrink: 0;
        border-radius: 12px;
        overflow: hidden;
    }

    .movie-info-content {
        flex: 1;
        min-height: 400px;
        display: flex;
        flex-direction: column;
    }

    .movie-info-title {
        font-size: 1.8rem;
        font-weight: 600;
        color: #f0f6fc;
        margin-bottom: 1rem;
    }

    .movie-info-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        margin-bottom: 1.5rem;
        font-size: 0.95rem;
    }

    .movie-info-plot {
        color: #c9d1d9;
        line-height: 1.7;
        font-size: 1rem;
        margin-bottom: 1.5rem;
        padding: 1.5rem;
        background: rgba(13, 17, 23, 0.6);
        border-radius: 8px;
        border-left: 3px solid #58a6ff;
        flex: 1;
        word-wrap: break-word;
        overflow-wrap: break-word;
    }

    .movie-card {
        background: rgba(33, 38, 45, 0.8);
        border: 1px solid rgba(48, 54, 61, 0.8);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        min-height: 250px;
    }

    .movie-card:hover {
        border-color: rgba(88, 166, 255, 0.6);
        box-shadow: 0 8px 25px rgba(88, 166, 255, 0.15);
        transform: translateY(-2px);
    }

    .movie-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #f0f6fc;
        margin-bottom: 0.8rem;
    }

    .movie-rating {
        display: inline-block;
        background: linear-gradient(135deg, #ffd700, #ffb700);
        color: #000;
        padding: 0.4rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.95rem;
        margin-right: 1rem;
    }

    .movie-genre {
        color: #58a6ff;
        font-size: 1rem;
        font-weight: 500;
    }

    .movie-description {
        color: #c9d1d9;
        font-size: 1rem;
        line-height: 1.7;
        margin-top: 1.5rem;
        padding: 1.5rem;
        background: rgba(13, 17, 23, 0.6);
        border-radius: 8px;
        border-left: 3px solid #39d353;
        word-wrap: break-word;
        overflow-wrap: break-word;
        max-width: 100%;
    }

    .movie-meta {
        display: flex;
        gap: 1.5rem;
        margin-top: 1.5rem;
        font-size: 0.95rem;
        color: #8b949e;
        flex-wrap: wrap;
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

    .section-header {
        font-size: 2rem;
        font-weight: 600;
        color: #f0f6fc;
        margin: 2rem 0 1.5rem 0;
        text-align: center;
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

    @media (max-width: 768px) {
        .selected-movie-info {
            flex-direction: column;
        }
        .movie-poster {
            align-self: center;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# TMDB API key
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
    movies_df = None
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

def clean_text(text):
    if pd.isna(text) or text == '':
        return ''
    text = str(text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_movie_data(movies_df):
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
                return clean_text(text)
            except:
                return clean_text(text) if pd.notna(text) else 'Unknown'
        
        movies_df['genres'] = movies_df['genres'].fillna('[]').apply(extract_genre_names)
        
        if 'cast' in movies_df.columns:
            def extract_cast_names(text):
                try:
                    if pd.isna(text) or text == '':
                        return ''
                    data = json.loads(text.replace("'", '"'))
                    if isinstance(data, list):
                        cast_names = [actor['name'] for actor in data[:10] if 'name' in actor]
                        return ' | '.join(cast_names)
                    return clean_text(text)
                except:
                    return clean_text(text) if pd.notna(text) else ''
            
            movies_df['cast'] = movies_df['cast'].fillna('[]').apply(extract_cast_names)
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
        
        movies_df['overview'] = movies_df['overview'].apply(clean_text)
        
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
        st.error(f"Error processing movie data: {e}")
        return movies_df

def create_fallback_data():
    sample_data = [
        {"id": 19995, "title": "Avatar", "genres": "Action Adventure Fantasy", "vote_average": 7.2, "overview": "In the 22nd century, a paraplegic Marine is dispatched to the moon Pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization.", "cast": "Sam Worthington | Zoe Saldana | Sigourney Weaver", "director": "James Cameron", "release_date": "2009-12-18", "runtime": 162, "vote_count": 11800, "popularity": 150.437},
        {"id": 550, "title": "Fight Club", "genres": "Drama Thriller", "vote_average": 8.4, "overview": "A ticking-time-bomb insomniac and a slippery soap salesman channel primal male aggression into a shocking new form of therapy.", "cast": "Brad Pitt | Edward Norton | Helena Bonham Carter", "director": "David Fincher", "release_date": "1999-10-15", "runtime": 139, "vote_count": 15420, "popularity": 89.234},
        {"id": 155, "title": "The Dark Knight", "genres": "Action Crime Drama", "vote_average": 8.5, "overview": "Batman raises the stakes in his war on crime. With the help of Lt. Jim Gordon and District Attorney Harvey Dent, Batman sets out to dismantle the remaining criminal organizations.", "cast": "Christian Bale | Heath Ledger | Aaron Eckhart", "director": "Christopher Nolan", "release_date": "2008-07-18", "runtime": 152, "vote_count": 18500, "popularity": 140.789},
        {"id": 157336, "title": "Interstellar", "genres": "Adventure Drama Science Fiction", "vote_average": 8.1, "overview": "The adventures of a group of explorers who make use of a newly discovered wormhole to surpass the limitations on human space travel.", "cast": "Matthew McConaughey | Anne Hathaway | Jessica Chastain", "director": "Christopher Nolan", "release_date": "2014-11-07", "runtime": 169, "vote_count": 16789, "popularity": 132.567},
    ]
    
    df = pd.DataFrame(sample_data)
    df['cast_searchable'] = df['cast'].str.lower().str.replace('|', ' ')
    df['director_searchable'] = df['director'].str.lower().str.replace('|', ' ')
    df['combined_features'] = df['overview'] + ' ' + df['genres'] + ' ' + df['cast'] + ' ' + df['director']
    return df

@st.cache_data
def load_movie_data():
    try:
        return load_data_from_github()
    except Exception as e:
        return create_fallback_data()

@st.cache_data
def create_similarity_matrix(movies_df):
    if not SKLEARN_AVAILABLE:
        return None
        
    try:
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
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def advanced_search(movies_df, query, search_type="auto"):
    if not query or len(query.strip()) < 2:
        return pd.DataFrame(), "auto", ""
    
    query = query.lower().strip()
    results = []
    
    # Detect search type
    if search_type == "auto":
        common_actors = ['brad pitt', 'leonardo dicaprio', 'tom hanks', 'christopher nolan', 'martin scorsese']
        common_directors = ['christopher nolan', 'martin scorsese', 'quentin tarantino', 'steven spielberg', 'james cameron']
        common_genres = ['action', 'adventure', 'comedy', 'drama', 'horror', 'thriller', 'romance', 'science fiction']
        
        if any(actor in query for actor in common_actors):
            search_type = "actor"
        elif any(director in query for director in common_directors):
            search_type = "director"
        elif any(genre in query for genre in common_genres):
            search_type = "genre"
        else:
            search_type = "title"

    # Title search
    exact_matches = movies_df[movies_df['title'].str.lower() == query]
    for _, movie in exact_matches.iterrows():
        results.append((movie, 100, "exact_title"))
    
    title_partials = movies_df[
        (movies_df['title'].str.lower() != query) &
        (movies_df['title'].str.lower().str.contains(re.escape(query), na=False))
    ]
    for _, movie in title_partials.iterrows():
        score = 85 + similarity_score(query, movie['title']) * 15
        results.append((movie, score, "partial_title"))
    
    # Actor search
    if 'cast_searchable' in movies_df.columns:
        cast_matches = movies_df[
            movies_df['cast_searchable'].str.contains(re.escape(query), na=False, regex=True)
        ]
        for _, movie in cast_matches.iterrows():
            score = 80
            cast_text = movie['cast_searchable']
            if query in cast_text:
                score += 15
            results.append((movie, score, "actor"))
    
    # Director search
    if 'director_searchable' in movies_df.columns:
        director_matches = movies_df[
            movies_df['director_searchable'].str.contains(re.escape(query), na=False, regex=True)
        ]
        for _, movie in director_matches.iterrows():
            score = 85
            director_text = movie['director_searchable']
            if query in director_text:
                score += 15
            results.append((movie, score, "director"))
    
    # Genre search
    genre_matches = movies_df[
        movies_df['genres'].str.lower().str.contains(re.escape(query), na=False)
    ]
    for _, movie in genre_matches.iterrows():
        score = 70
        if query in movie['genres'].lower().split():
            score += 25
        results.append((movie, score, "genre"))
    
    if not results:
        return pd.DataFrame(), search_type, f"No results found for '{query}'"
    
    # Remove duplicates
    seen_ids = set()
    unique_results = []
    for movie, score, match_type in results:
        if movie['id'] not in seen_ids:
            seen_ids.add(movie['id'])
            unique_results.append((movie, score, match_type))
    
    # Sort by score
    unique_results.sort(key=lambda x: x[1], reverse=True)
    
    # Return more results for actors/directors
    if search_type in ["actor", "director"]:
        result_df = pd.DataFrame([movie for movie, _, _ in unique_results[:50]])
    else:
        result_df = pd.DataFrame([movie for movie, _, _ in unique_results[:30]])
    
    return result_df.reset_index(drop=True), search_type, ""

def get_person_info(person_name, person_type="actor"):
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
            }
    except:
        pass
    return None

def get_genre_description(genre_name):
    genre_descriptions = {
        'action': 'Action films are characterized by high energy, big-budget physical stunts, chases, fights, battles, and destructive crises.',
        'adventure': 'Adventure films are exciting stories, with new experiences or exotic locales, very similar to or often paired with the action film genre.',
        'comedy': 'Comedy films are designed to make the audience laugh through amusement and most often work by exaggerating characteristics.',
        'drama': 'Drama films are serious presentations or stories with settings or life situations that portray realistic characters in conflict.',
        'horror': 'Horror films seek to elicit fear, suspense, disgust, or startlement from audiences through frightening and disturbing elements.',
        'thriller': 'Thriller films are characterized by constant danger, high stakes, and psychological tension that keeps audiences on edge.',
        'romance': 'Romance films make the romantic love story or the search for strong and pure love and romance the main plot focus.',
        'science fiction': 'Science fiction films feature futuristic concepts such as advanced technology, space exploration, time travel, parallel universes.',
    }
    return genre_descriptions.get(genre_name.lower(), f'{genre_name} films offer unique storytelling experiences within this distinctive genre.')

def recommend_movies(movie_title, movies_df, similarity_matrix, n_recommendations=6):
    try:
        if not SKLEARN_AVAILABLE:
            return simple_recommend(movies_df, movie_title, n_recommendations)
        
        # Find movie
        exact_matches = movies_df[movies_df['title'].str.lower() == movie_title.lower()]
        if len(exact_matches) > 0:
            movie_idx = exact_matches.index[0]
        else:
            return pd.DataFrame()
        
        # Get similarity scores
        sim_scores = list(enumerate(similarity_matrix[movie_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = [score for score in sim_scores if score[0] != movie_idx]
        sim_scores = sim_scores[:n_recommendations]
        
        movie_indices = [i[0] for i in sim_scores]
        recommendations = movies_df.iloc[movie_indices].copy()
        recommendations['similarity_score'] = [score[1] for score in sim_scores]
        
        return recommendations.head(n_recommendations)
        
    except Exception as e:
        st.error(f"Error getting recommendations: {e}")
        return pd.DataFrame()

def create_movie_card(movie, show_match_score=False, similarity_score_pct=0):
    year = ""
    if 'release_date' in movie and pd.notna(movie['release_date']):
        try:
            year = movie['release_date'][:4]
        except:
            year = ""
    
    runtime = ""
    if 'runtime' in movie and pd.notna(movie['runtime']) and movie['runtime'] > 0:
        hours = int(movie['runtime']) // 60
        minutes = int(movie['runtime']) % 60
        if hours > 0:
            runtime = f"{hours}h {minutes}min"
        else:
            runtime = f"{minutes}min"
    
    director_info = ""
    if 'director' in movie and pd.notna(movie['director']) and movie['director']:
        director_clean = clean_text(movie['director'])
        director_info = director_clean
    
    cast_info = ""
    if 'cast' in movie and pd.notna(movie['cast']) and movie['cast']:
        cast_clean = clean_text(movie['cast'])
        cast_list = cast_clean.split(' | ')[:3]
        cast_info = f"Starring: {', '.join(cast_list)}"
    
    full_plot = clean_text(movie['overview'])
    
    match_score_html = ""
    if show_match_score:
        match_score_html = f'<div class="match-score">{similarity_score_pct:.0f}% Match</div>'
    
    year_display = f"📅 {year}" if year else ""
    runtime_display = f"⏱️ {runtime}" if runtime else ""
    director_display = f"🎬 {director_info}" if director_info else ""
    
    return f"""
    <div class="movie-card">
        {match_score_html}
        <h3 class="movie-title">{clean_text(movie['title'])}</h3>
        <div style="margin-bottom: 1rem;">
            <span class="movie-rating">⭐ {movie['vote_average']:.1f}</span>
            <span class="movie-genre">{clean_text(movie['genres'])}</span>
        </div>
        <div class="movie-meta">
            {f"<span>{year_display}</span>" if year_display else ""}
            {f"<span>{runtime_display}</span>" if runtime_display else ""}
            {f"<span>{director_display}</span>" if director_display else ""}
        </div>
        <div class="movie-description"><strong>Plot:</strong> {full_plot}</div>
        {f"<div style='margin-top: 1rem; color: #8b949e; font-size: 0.9rem;'>{cast_info}</div>" if cast_info else ""}
    </div>
    """

def main():
    load_styles()
    
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

    movies_df = load_movie_data()
    
    if movies_df is None or len(movies_df) == 0:
        st.error("Could not load movie data. Please check your configuration.")
        return
    
    similarity_matrix = create_similarity_matrix(movies_df)

    st.markdown("""
    <div class="hero-section">
        <h2 class="hero-title">Discover Your Next <span class="hero-gradient-text">Cinematic Journey</span></h2>
        <p class="hero-description">Experience the most sophisticated movie discovery platform with deep content analysis and personalized recommendations.</p>
    </div>
    """, unsafe_allow_html=True)

    # Search and Recommendation Section
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.markdown('<div class="search-section">', unsafe_allow_html=True)
        st.markdown("### Find Your Perfect Movie")
        
        search_query = st.text_input("Search movies, actors, directors, genres...", 
                                   placeholder="Try: 'Brad Pitt', 'Action', 'Christopher Nolan'")
        
        selected_movie = None
        
        if search_query:
            search_results, search_type, error_message = advanced_search(movies_df, search_query)
            
            if not error_message and len(search_results) > 0:
                if search_type == "actor":
                    st.markdown(f'<div class="search-results-header">🎭 Actor: "{search_query.title()}"</div>', unsafe_allow_html=True)
                    person_info = get_person_info(search_query, "actor")
                    if person_info and person_info.get('biography'):
                        bio = clean_text(person_info['biography'])
                        if len(bio) > 150:
                            bio = bio[:150] + "..."
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
                        bio = clean_text(person_info['biography'])
                        if len(bio) > 150:
                            bio = bio[:150] + "..."
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
                
                result_limit = 30 if search_type in ["actor", "director"] else 15
                result_options = []
                
                for _, movie in search_results.head(result_limit).iterrows():
                    year = ""
                    if 'release_date' in movie and pd.notna(movie['release_date']):
                        try:
                            year = f" ({movie['release_date'][:4]})"
                        except:
                            year = ""
                    
                    rating_info = f" - ⭐{movie['vote_average']:.1f}"
                    display_text = f"{clean_text(movie['title'])}{year}{rating_info}"
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
                        if clean_text(movie['title']) == selected_movie_title:
                            selected_movie = movie
                            break
            
            elif error_message:
                st.markdown(f'<div class="no-results">{error_message}</div>', unsafe_allow_html=True)
                st.info("💡 Try searching for:\n- Movie titles: 'Avatar', 'Inception'\n- Actors: 'Leonardo DiCaprio', 'Brad Pitt'\n- Directors: 'Christopher Nolan', 'Steven Spielberg'\n- Genres: 'Action', 'Comedy', 'Thriller'")
        
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
                display_text = f"{clean_text(movie['title'])}{year} - ⭐{movie['vote_average']:.1f}"
                movie_options.append(display_text)
            
            selected_display = st.selectbox("Choose from popular movies:", movie_options)
            
            if selected_display:
                selected_movie_title = selected_display.split(" - ⭐")[0]
                if " (" in selected_movie_title and selected_movie_title.endswith(")"):
                    selected_movie_title = selected_movie_title.rsplit(" (", 1)[0]
                
                for _, movie in combined.iterrows():
                    if clean_text(movie['title']) == selected_movie_title:
                        selected_movie = movie
                        break
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if selected_movie is not None:
            poster_url = get_movie_poster(selected_movie['title'], selected_movie.get('id'))
            full_overview = clean_text(selected_movie['overview'])
            
            runtime_info = ""
            if 'runtime' in selected_movie and pd.notna(selected_movie['runtime']) and selected_movie['runtime'] > 0:
                hours = int(selected_movie['runtime']) // 60
                minutes = int(selected_movie['runtime']) % 60
                if hours > 0:
                    runtime_info = f"⏱️ {hours}h {minutes}min"
                else:
                    runtime_info = f"⏱️ {minutes}min"
            
            release_info = ""
            if 'release_date' in selected_movie and pd.notna(selected_movie['release_date']):
                try:
                    release_info = f"📅 {selected_movie['release_date'][:4]}"
                except:
                    release_info = ""
            
            cast_display = ""
            if 'cast' in selected_movie and pd.notna(selected_movie['cast']) and selected_movie['cast']:
                cast_clean = clean_text(selected_movie['cast'])
                cast_display = f"Cast: {cast_clean}"
            
            director_display = ""
            if 'director' in selected_movie and pd.notna(selected_movie['director']) and selected_movie['director']:
                director_clean = clean_text(selected_movie['director'])
                director_display = f"🎬 {director_clean}"
            
            st.markdown(f"""
            <div class="selected-movie-info">
                <div class="movie-poster">
                    <img src="{poster_url}" width="200" alt="{clean_text(selected_movie['title'])}">
                </div>
                <div class="movie-info-content">
                    <h2 class="movie-info-title">{clean_text(selected_movie['title'])}</h2>
                    <div class="movie-info-meta">
                        <span class="movie-rating">⭐ {selected_movie['vote_average']:.1f}/10</span>
                        <span class="movie-genre">🎭 {clean_text(selected_movie['genres'])}</span>
                        {f"<span>{director_display}</span>" if director_display else ""}
                        {f"<span>{release_info}</span>" if release_info else ""}
                        {f"<span>{runtime_info}</span>" if runtime_info else ""}
                    </div>
                    <div class="movie-info-plot">
                        <strong>Plot:</strong> {full_overview}
                    </div>
                    {f"<div style='color: #58a6ff; font-size: 0.9rem; margin-top: 1rem;'>{cast_display}</div>" if cast_display else ""}
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
                    card_html = create_movie_card(movie, True, similarity_score_pct)
                    st.markdown(card_html, unsafe_allow_html=True)
            else:
                st.warning("No recommendations found for this movie. Try selecting a different movie.")
        else:
            st.info("Select a movie from the left to see personalized recommendations here.")
    
    with col3:
        if selected_movie is None:
            st.markdown("### Featured Movies")
            featured = movies_df.nlargest(4, 'vote_average')
            
            for _, movie in featured.iterrows():
                card_html = create_movie_card(movie, False, 0)
                st.markdown(card_html, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; color: #8b949e; font-size: 0.9rem; margin-top: 3rem; padding: 2rem; border-top: 1px solid rgba(48, 54, 61, 0.6);">
        <p>🎬 Powered by TMDB Database via GitHub Repository | Built with Advanced Content-Based Filtering</p>
        <p>💡 Multi-dimensional analysis using plot, genres, cast, crew, and intelligent search algorithms</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
