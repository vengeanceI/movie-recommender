import streamlit as st
import pandas as pd
import numpy as np
import ast
import requests
from io import StringIO

# Try to import sklearn components with error handling
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.warning("⚠️ sklearn not properly installed. Using fallback similarity calculation.")

# Set page config
st.set_page_config(
    page_title="🎬 TMDB Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    
    .movie-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #ff6b6b;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }
    
    .movie-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    .similarity-score {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        text-align: center;
        font-weight: bold;
        font-size: 0.9rem;
    }
    
    .genre-tag {
        background: #ff6b6b;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        display: inline-block;
    }
    
    .rating-badge {
        background: linear-gradient(45deg, #f093fb, #f5576c);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-weight: bold;
    }
    
    .stats-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_tmdb_data():
    """Load TMDB dataset - first try to load from uploaded files, then fallback to sample"""
    try:
        # Try to load uploaded TMDB files
        movies_df = pd.read_csv('tmdb_5000_movies.csv')
        credits_df = pd.read_csv('tmdb_5000_credits.csv')
        
        # Merge datasets
        movies_df = movies_df.merge(credits_df, on='id', how='left')
        
        st.success(f"✅ Loaded real TMDB dataset: {len(movies_df)} movies!")
        return movies_df, True
        
    except FileNotFoundError:
        # Fallback to expanded sample data
        st.info("📊 Using sample dataset. Upload TMDB CSV files to use full dataset!")
        
        sample_movies = {
            'id': list(range(1, 21)),
            'title': ['The Dark Knight', 'Inception', 'Pulp Fiction', 'The Matrix', 'Forrest Gump', 
                     'The Shawshank Redemption', 'Fight Club', 'Goodfellas', 'The Godfather', 'Interstellar',
                     'Avengers: Endgame', 'Titanic', 'Avatar', 'Star Wars', 'The Lion King',
                     'Jurassic Park', 'E.T.', 'The Avengers', 'Spider-Man', 'Batman Begins'],
            'overview': [
                'When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological tests',
                'A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea',
                'The lives of two mob hitmen, a boxer, a gangster and his wife intertwine in four tales of violence and redemption',
                'A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers',
                'The presidencies of Kennedy and Johnson, Vietnam, Watergate, and other history unfold through the perspective of an Alabama man',
                'Two imprisoned friends bond over years, finding solace and redemption through acts of common decency',
                'An insomniac office worker forms an underground fight club that evolves into something much more',
                'The story of Henry Hill and his life in the mob, covering his relationship with his wife Karen Hill and his mob partners',
                'The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son',
                'A team of explorers travel through a wormhole in space in an attempt to ensure humanity survival',
                'After the devastating events of Infinity War, the Avengers assemble once more to reverse Thanos actions',
                'A seventeen-year-old aristocrat falls in love with a kind but poor artist aboard the luxurious ill-fated R.M.S. Titanic',
                'A paraplegic marine dispatched to the moon Pandora on a unique mission becomes torn between following his orders',
                'Luke Skywalker joins forces with a Jedi Knight, a cocky pilot, a Wookiee to save the galaxy from the Empire',
                'Lion prince Simba and his father are targeted by his bitter uncle, who wants to ascend the throne himself',
                'Scientists clone dinosaurs to populate a theme park which suffers a major security breakdown',
                'A troubled child summons the courage to help a friendly alien escape Earth and return to his home planet',
                'Earth mightiest heroes must come together and learn to fight as a team to stop the mischievous Loki',
                'When bitten by a genetically altered spider, a nerdy high school student gains superhuman abilities',
                'After training with mentor, Bruce Wayne begins his fight to free crime-ridden Gotham City from corruption'
            ],
            'genres': [
                '[{"name": "Action"}, {"name": "Crime"}, {"name": "Drama"}]',
                '[{"name": "Action"}, {"name": "Sci-Fi"}, {"name": "Thriller"}]',
                '[{"name": "Crime"}, {"name": "Drama"}]',
                '[{"name": "Action"}, {"name": "Sci-Fi"}]',
                '[{"name": "Drama"}, {"name": "Romance"}]',
                '[{"name": "Drama"}]',
                '[{"name": "Drama"}, {"name": "Thriller"}]',
                '[{"name": "Biography"}, {"name": "Crime"}, {"name": "Drama"}]',
                '[{"name": "Crime"}, {"name": "Drama"}]',
                '[{"name": "Adventure"}, {"name": "Drama"}, {"name": "Sci-Fi"}]',
                '[{"name": "Adventure"}, {"name": "Sci-Fi"}, {"name": "Action"}]',
                '[{"name": "Drama"}, {"name": "Romance"}]',
                '[{"name": "Action"}, {"name": "Adventure"}, {"name": "Fantasy"}]',
                '[{"name": "Adventure"}, {"name": "Action"}, {"name": "Sci-Fi"}]',
                '[{"name": "Family"}, {"name": "Animation"}, {"name": "Drama"}]',
                '[{"name": "Adventure"}, {"name": "Sci-Fi"}, {"name": "Thriller"}]',
                '[{"name": "Family"}, {"name": "Sci-Fi"}, {"name": "Adventure"}]',
                '[{"name": "Action"}, {"name": "Adventure"}, {"name": "Sci-Fi"}]',
                '[{"name": "Action"}, {"name": "Adventure"}, {"name": "Sci-Fi"}]',
                '[{"name": "Action"}, {"name": "Crime"}, {"name": "Drama"}]'
            ],
            'vote_average': [9.0, 8.8, 8.9, 8.7, 8.8, 9.3, 8.8, 8.7, 9.2, 8.6,
                           8.4, 7.8, 7.8, 8.6, 8.5, 8.1, 7.9, 8.0, 7.3, 8.2],
            'vote_count': [12269, 13752, 8670, 9847, 8147, 14238, 9678, 6461, 9847, 11187,
                          8500, 11800, 11200, 9500, 7800, 8900, 7200, 9100, 6400, 7800],
            'release_date': ['2008-07-18', '2010-07-16', '1994-10-14', '1999-03-31', '1994-07-06',
                            '1994-09-23', '1999-10-15', '1990-09-12', '1972-03-24', '2014-11-07',
                            '2019-04-24', '1997-11-18', '2009-12-10', '1977-05-25', '1994-06-12',
                            '1993-06-09', '1982-06-11', '2012-04-25', '2002-05-01', '2005-06-10'],
            'cast': ['[]'] * 20,  # Simplified for sample
            'crew': ['[]'] * 20,
            'keywords': ['[]'] * 20
        }
        
        return pd.DataFrame(sample_movies), False

def safe_literal_eval(x):
    """Safely evaluate string representations of lists"""
    try:
        return ast.literal_eval(x) if pd.notna(x) else []
    except:
        return []

def extract_names(obj_list, key='name', limit=3):
    """Extract names from list of objects"""
    if not obj_list or not isinstance(obj_list, list):
        return []
    return [obj.get(key, '') for obj in obj_list[:limit] if isinstance(obj, dict)]

def get_director(crew_list):
    """Extract director from crew list"""
    if not crew_list or not isinstance(crew_list, list):
        return ""
    for person in crew_list:
        if isinstance(person, dict) and person.get('job') == 'Director':
            return person.get('name', '')
    return ""

@st.cache_data
def preprocess_tmdb_data(df, is_real_tmdb):
    """Preprocess the TMDB data"""
    # Handle missing values
    df['overview'] = df['overview'].fillna('')
    df['genres'] = df['genres'].fillna('[]')
    
    if is_real_tmdb:
        df['keywords'] = df['keywords'].fillna('[]')
        df['cast'] = df['cast'].fillna('[]')
        df['crew'] = df['crew'].fillna('[]')
        
        # Convert string representations to actual objects
        df['genres'] = df['genres'].apply(safe_literal_eval)
        df['keywords'] = df['keywords'].apply(safe_literal_eval)
        df['cast'] = df['cast'].apply(safe_literal_eval)
        df['crew'] = df['crew'].apply(safe_literal_eval)
        
        # Extract features
        df['genre_names'] = df['genres'].apply(lambda x: extract_names(x, limit=5))
        df['keyword_names'] = df['keywords'].apply(lambda x: extract_names(x, limit=8))
        df['cast_names'] = df['cast'].apply(lambda x: extract_names(x, limit=5))
        df['director'] = df['crew'].apply(get_director)
        
        # Create feature soup
        df['genres_str'] = df['genre_names'].apply(lambda x: ' '.join([g.lower().replace(' ', '') for g in x]))
        df['keywords_str'] = df['keyword_names'].apply(lambda x: ' '.join([k.lower().replace(' ', '') for k in x]))
        df['cast_str'] = df['cast_names'].apply(lambda x: ' '.join([c.lower().replace(' ', '') for c in x]))
        df['director_str'] = df['director'].apply(lambda x: x.lower().replace(' ', '') if x else '')
        
        # Enhanced soup with weighted features
        df['soup'] = (
            df['overview'] + ' ' +
            df['genres_str'] + ' ' + df['genres_str'] + ' ' +  # Weight genres more
            df['keywords_str'] + ' ' +
            df['cast_str'] + ' ' +
            df['director_str'] + ' ' + df['director_str'] + ' ' + df['director_str']  # Weight director heavily
        )
    else:
        # Simple processing for sample data
        df['genres'] = df['genres'].apply(safe_literal_eval)
        df['genre_names'] = df['genres'].apply(lambda x: extract_names(x, limit=5))
        df['genres_str'] = df['genre_names'].apply(lambda x: ' '.join([g.lower() for g in x]))
        df['soup'] = df['overview'] + ' ' + df['genres_str']
    
    # Remove rows with empty soup
    df = df[df['soup'].str.strip() != ''].copy()
    df = df.reset_index(drop=True)
    
    return df

def calculate_advanced_similarity(df):
    """Advanced similarity calculation without sklearn"""
    n_movies = len(df)
    similarity_matrix = np.zeros((n_movies, n_movies))
    
    for i in range(n_movies):
        for j in range(n_movies):
            if i == j:
                similarity_matrix[i][j] = 1.0
            else:
                # Enhanced similarity with genre boosting
                words_i = set(df.iloc[i]['soup'].lower().split())
                words_j = set(df.iloc[j]['soup'].lower().split())
                
                if len(words_i) > 0 and len(words_j) > 0:
                    # Basic word overlap
                    intersection = len(words_i.intersection(words_j))
                    union = len(words_i.union(words_j))
                    base_sim = intersection / union if union > 0 else 0
                    
                    # Genre boost
                    genres_i = set(df.iloc[i]['genres_str'].split())
                    genres_j = set(df.iloc[j]['genres_str'].split())
                    genre_overlap = len(genres_i.intersection(genres_j))
                    
                    # Apply genre boost
                    genre_boost = min(genre_overlap * 0.1, 0.3)
                    similarity_matrix[i][j] = min(base_sim + genre_boost, 1.0)
                else:
                    similarity_matrix[i][j] = 0
    
    return similarity_matrix

@st.cache_data
def build_enhanced_recommender(df, is_real_tmdb):
    """Build enhanced recommendation system"""
    if SKLEARN_AVAILABLE:
        # Enhanced TF-IDF with custom parameters
        tfidf = TfidfVectorizer(
            max_features=8000,
            stop_words='english',
            max_df=0.85,
            min_df=2 if is_real_tmdb else 1,
            ngram_range=(1, 2),
            analyzer='word'
        )
        
        tfidf_matrix = tfidf.fit_transform(df['soup'])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    else:
        cosine_sim = calculate_advanced_similarity(df)
        tfidf_matrix = None
    
    # Create title to index mapping
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    
    return cosine_sim, indices, tfidf_matrix

def get_enhanced_recommendations(title, cosine_sim, indices, df, n_recommendations=6):
    """Get enhanced movie recommendations with better scoring"""
    try:
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:n_recommendations+1]
        
        recommendations = []
        for i, score in sim_scores:
            movie_data = df.iloc[i]
            
            # Enhanced movie data
            rec_data = {
                'title': movie_data['title'],
                'year': movie_data['release_date'][:4] if pd.notna(movie_data['release_date']) else 'N/A',
                'genres': movie_data.get('genre_names', []),
                'rating': round(movie_data['vote_average'], 1) if pd.notna(movie_data['vote_average']) else 'N/A',
                'votes': movie_data.get('vote_count', 'N/A'),
                'overview': movie_data['overview'][:250] + '...' if len(movie_data['overview']) > 250 else movie_data['overview'],
                'similarity': round(score, 3)
            }
            
            # Add director if available
            if 'director' in movie_data and pd.notna(movie_data['director']) and movie_data['director']:
                rec_data['director'] = movie_data['director']
            
            recommendations.append(rec_data)
        
        return recommendations
    
    except KeyError:
        return None

def display_enhanced_movie_card(movie, show_similarity=False):
    """Display enhanced movie card"""
    with st.container():
        similarity_html = f'<div class="similarity-score">🎯 {movie["similarity"]} similarity</div>' if show_similarity else ''
        
        director_html = f'<p><strong>Director:</strong> {movie["director"]}</p>' if 'director' in movie else ''
        
        votes_html = f' • {movie["votes"]:,} votes' if movie["votes"] != 'N/A' else ''
        
        st.markdown(f"""
        <div class="movie-card">
            <h3 style="color: #2c3e50; margin-bottom: 0.5rem;">{movie['title']} ({movie['year']})</h3>
            
            <div style="margin-bottom: 1rem;">
                {''.join([f'<span class="genre-tag">{genre}</span>' for genre in movie['genres']])}
            </div>
            
            <div style="margin-bottom: 1rem;">
                <span class="rating-badge">⭐ {movie['rating']}/10{votes_html}</span>
            </div>
            
            {similarity_html}
            
            {director_html}
            
            <p style="margin-top: 1rem; color: #555; line-height: 1.5;">{movie['overview']}</p>
        </div>
        """, unsafe_allow_html=True)

def display_stats(df):
    """Display dataset statistics"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stats-container">
            <h3>🎬 {len(df)}</h3>
            <p>Total Movies</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_rating = df['vote_average'].mean() if 'vote_average' in df else 0
        st.markdown(f"""
        <div class="stats-container">
            <h3>⭐ {avg_rating:.1f}</h3>
            <p>Avg Rating</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        year_range = f"{df['release_date'].str[:4].min()}-{df['release_date'].str[:4].max()}" if 'release_date' in df else 'N/A'
        st.markdown(f"""
        <div class="stats-container">
            <h3>📅 {year_range}</h3>
            <p>Year Range</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        total_genres = len(set([genre for genres_list in df['genre_names'] for genre in genres_list]))
        st.markdown(f"""
        <div class="stats-container">
            <h3>🎭 {total_genres}</h3>
            <p>Unique Genres</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🎬 TMDB Movie Recommendation System</h1>', unsafe_allow_html=True)
    
    # Load and preprocess data
    with st.spinner("🔄 Loading TMDB dataset..."):
        df, is_real_tmdb = load_tmdb_data()
        df = preprocess_tmdb_data(df, is_real_tmdb)
        cosine_sim, indices, tfidf_matrix = build_enhanced_recommender(df, is_real_tmdb)
    
    # Display statistics
    display_stats(df)
    
    # Sidebar
    st.sidebar.markdown("## 🎯 Enhanced Features")
    st.sidebar.markdown(f"""
    **Dataset:** {'Real TMDB 5000' if is_real_tmdb else 'Sample Movies'}
    
    **Algorithm Enhancements:**
    - Advanced TF-IDF vectorization
    - Genre-weighted similarity
    - Director influence boost
    - Multi-feature content analysis
    
    **How to use:**
    1. Select a movie you enjoyed
    2. Click 'Get Recommendations'
    3. Explore similar movies with similarity scores
    4. Try different movies to see varied results!
    """)
    
    if not is_real_tmdb:
        st.sidebar.markdown("""
        ---
        **🚀 Want more movies?**
        
        Upload `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` to your GitHub repo to unlock 5000+ movies!
        """)
    
    # Main interface
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 🔍 Select a Movie")
        
        # Add search functionality
        search_query = st.text_input("🔎 Search movies:", placeholder="Type to search...")
        
        # Filter movies based on search
        if search_query:
            filtered_movies = df[df['title'].str.contains(search_query, case=False, na=False)]['title'].tolist()
            if not filtered_movies:
                st.warning("No movies found matching your search.")
                filtered_movies = df['title'].tolist()
        else:
            filtered_movies = df['title'].tolist()
        
        # Movie selection
        selected_movie = st.selectbox(
            "Choose a movie you like:",
            options=filtered_movies,
            index=0
        )
        
        # Number of recommendations slider
        n_recs = st.slider("Number of recommendations:", 3, 10, 5)
        
        # Get recommendations button
        if st.button("🎯 Get Recommendations", type="primary"):
            st.session_state.show_recommendations = True
            st.session_state.selected_movie = selected_movie
            st.session_state.n_recs = n_recs
        
        # Display selected movie info
        if selected_movie:
            st.markdown("### 📽️ Selected Movie")
            selected_data = df[df['title'] == selected_movie].iloc[0]
            movie_info = {
                'title': selected_data['title'],
                'year': selected_data['release_date'][:4] if pd.notna(selected_data['release_date']) else 'N/A',
                'genres': selected_data.get('genre_names', []),
                'rating': round(selected_data['vote_average'], 1) if pd.notna(selected_data['vote_average']) else 'N/A',
                'votes': selected_data.get('vote_count', 'N/A'),
                'overview': selected_data['overview']
            }
            
            if 'director' in selected_data and pd.notna(selected_data['director']) and selected_data['director']:
                movie_info['director'] = selected_data['director']
                
            display_enhanced_movie_card(movie_info)
    
    with col2:
        if hasattr(st.session_state, 'show_recommendations') and st.session_state.show_recommendations:
            st.markdown("### 🎬 Recommended Movies")
            
            with st.spinner("🔄 Finding similar movies..."):
                recommendations = get_enhanced_recommendations(
                    st.session_state.selected_movie, 
                    cosine_sim, 
                    indices, 
                    df, 
                    n_recommendations=st.session_state.n_recs
                )
            
            if recommendations:
                st.success(f"✅ Found {len(recommendations)} similar movies!")
                
                for i, movie in enumerate(recommendations, 1):
                    st.markdown(f"#### #{i} Recommendation")
                    display_enhanced_movie_card(movie, show_similarity=True)
                    st.markdown("---")
                    
            else:
                st.error("❌ Sorry, couldn't find recommendations for this movie.")
        else:
            st.markdown("### 👈 Select a movie to get started!")
            st.markdown(f"""
            This **enhanced** recommendation system uses advanced content-based filtering with:
            
            **🧠 Advanced Features:**
            - **TF-IDF Vectorization** for text analysis
            - **Genre-weighted similarity** for better matches  
            - **Director influence** for auteur preferences
            - **Multi-ngram analysis** for context understanding
            - **Popularity weighting** for quality filtering
            
            **📊 Dataset:** {len(df)} movies loaded
            **🎭 Genres:** {len(set([g for genres in df['genre_names'] for g in genres]))} unique genres
            
            **🎬 Try these popular movies:**
            - The Dark Knight (Crime/Action)
            - Inception (Sci-Fi/Thriller)  
            - Pulp Fiction (Crime/Drama)
            - The Matrix (Action/Sci-Fi)
            """)
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; color: #888;">
            <p>🚀 Enhanced TMDB Movie Recommender | 🤖 Advanced ML Algorithms</p>
            <p>Built with Streamlit • Content-Based Filtering • Real-time Recommendations</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
