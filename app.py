import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import requests
import pickle
from io import StringIO

# Set page config
st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
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
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ff6b6b;
        margin-bottom: 1rem;
    }
    
    .similarity-score {
        background: #e3f2fd;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }
    
    .genre-tag {
        background: #ff6b6b;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin-right: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_sample_data():
    """Load sample movie data (replace with actual TMDB data later)"""
    sample_movies = {
        'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'title': ['The Dark Knight', 'Inception', 'Pulp Fiction', 'The Matrix', 'Forrest Gump', 
                 'The Shawshank Redemption', 'Fight Club', 'Goodfellas', 'The Godfather', 'Interstellar'],
        'overview': [
            'When the menace known as the Joker wreaks havoc and chaos on the people of Gotham',
            'A thief who steals corporate secrets through the use of dream-sharing technology',
            'The lives of two mob hitmen, a boxer, a gangster and his wife intertwine',
            'A computer hacker learns from mysterious rebels about the true nature of his reality',
            'The presidencies of Kennedy and Johnson through the eyes of an Alabama man',
            'Two imprisoned friends bond over years, finding solace and redemption',
            'An insomniac office worker forms an underground fight club',
            'The story of Henry Hill and his life in the mob',
            'The aging patriarch of a crime dynasty transfers control to his reluctant son',
            'A team of explorers travel through a wormhole in space'
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
            '[{"name": "Adventure"}, {"name": "Drama"}, {"name": "Sci-Fi"}]'
        ],
        'vote_average': [9.0, 8.8, 8.9, 8.7, 8.8, 9.3, 8.8, 8.7, 9.2, 8.6],
        'release_date': ['2008-07-18', '2010-07-16', '1994-10-14', '1999-03-31', '1994-07-06',
                        '1994-09-23', '1999-10-15', '1990-09-12', '1972-03-24', '2014-11-07']
    }
    return pd.DataFrame(sample_movies)

def safe_literal_eval(x):
    """Safely evaluate string representations of lists"""
    try:
        return ast.literal_eval(x) if pd.notna(x) else []
    except:
        return []

def extract_genre_names(genres_str):
    """Extract genre names from JSON-like string"""
    genres = safe_literal_eval(genres_str)
    return [genre.get('name', '') for genre in genres if isinstance(genre, dict)]

@st.cache_data
def preprocess_data(df):
    """Preprocess the movie data"""
    # Extract genres
    df['genre_list'] = df['genres'].apply(extract_genre_names)
    df['genres_str'] = df['genre_list'].apply(lambda x: ' '.join([g.lower() for g in x]))
    
    # Create content soup for recommendation
    df['soup'] = df['overview'].fillna('') + ' ' + df['genres_str']
    
    # Remove rows with empty soup
    df = df[df['soup'].str.strip() != '']
    
    return df

@st.cache_data
def build_recommender(df):
    """Build the recommendation system"""
    # Create TF-IDF matrix
    tfidf = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        max_df=0.8,
        min_df=1,
        ngram_range=(1, 2)
    )
    
    tfidf_matrix = tfidf.fit_transform(df['soup'])
    
    # Calculate cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Create title to index mapping
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    
    return cosine_sim, indices, tfidf_matrix

def get_recommendations(title, cosine_sim, indices, df, n_recommendations=6):
    """Get movie recommendations"""
    try:
        # Get the index of the movie
        idx = indices[title]
        
        # Get similarity scores
        sim_scores = list(enumerate(cosine_sim[idx]))
        
        # Sort by similarity
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N similar movies (excluding the input movie)
        sim_scores = sim_scores[1:n_recommendations+1]
        
        # Get movie indices
        movie_indices = [i[0] for i in sim_scores]
        
        # Return recommendations
        recommendations = []
        for i, score in sim_scores:
            movie_data = df.iloc[i]
            recommendations.append({
                'title': movie_data['title'],
                'year': movie_data['release_date'][:4] if pd.notna(movie_data['release_date']) else 'N/A',
                'genres': movie_data['genre_list'],
                'rating': movie_data['vote_average'],
                'overview': movie_data['overview'][:200] + '...' if len(movie_data['overview']) > 200 else movie_data['overview'],
                'similarity': round(score, 3)
            })
        
        return recommendations
    
    except KeyError:
        return None

def display_movie_card(movie, show_similarity=False):
    """Display a movie card"""
    with st.container():
        st.markdown(f"""
        <div class="movie-card">
            <h3 style="color: #2c3e50;">{movie['title']} ({movie['year']})</h3>
            <div style="margin-bottom: 10px;">
                {''.join([f'<span class="genre-tag">{genre}</span>' for genre in movie['genres']])}
            </div>
            <p><strong>Rating:</strong> ⭐ {movie['rating']}/10</p>
            {f'<div class="similarity-score">Similarity Score: {movie["similarity"]}</div>' if show_similarity else ''}
            <p style="margin-top: 10px; color: #555;">{movie['overview']}</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🎬 Movie Recommendation System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("## 🎯 How to Use")
    st.sidebar.markdown("""
    1. **Select a movie** from the dropdown
    2. **Click 'Get Recommendations'**
    3. **Explore** similar movies!
    
    ---
    
    ### 📊 Features:
    - Content-based filtering
    - Genre-aware recommendations
    - Similarity scoring
    - Interactive interface
    """)
    
    # Load and preprocess data
    with st.spinner("Loading movie database..."):
        df = load_sample_data()
        df = preprocess_data(df)
        cosine_sim, indices, tfidf_matrix = build_recommender(df)
    
    st.success(f"✅ Loaded {len(df)} movies successfully!")
    
    # Main interface
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 🔍 Select a Movie")
        
        # Movie selection
        selected_movie = st.selectbox(
            "Choose a movie you like:",
            options=df['title'].tolist(),
            index=0
        )
        
        # Get recommendations button
        if st.button("🎯 Get Recommendations", type="primary"):
            st.session_state.show_recommendations = True
            st.session_state.selected_movie = selected_movie
        
        # Display selected movie info
        if selected_movie:
            st.markdown("### 📽️ Selected Movie")
            selected_data = df[df['title'] == selected_movie].iloc[0]
            movie_info = {
                'title': selected_data['title'],
                'year': selected_data['release_date'][:4],
                'genres': selected_data['genre_list'],
                'rating': selected_data['vote_average'],
                'overview': selected_data['overview']
            }
            display_movie_card(movie_info)
    
    with col2:
        if hasattr(st.session_state, 'show_recommendations') and st.session_state.show_recommendations:
            st.markdown("### 🎬 Recommended Movies")
            
            with st.spinner("Finding similar movies..."):
                recommendations = get_recommendations(
                    st.session_state.selected_movie, 
                    cosine_sim, 
                    indices, 
                    df, 
                    n_recommendations=5
                )
            
            if recommendations:
                st.success(f"Found {len(recommendations)} similar movies!")
                
                for i, movie in enumerate(recommendations, 1):
                    st.markdown(f"#### #{i} Recommendation")
                    display_movie_card(movie, show_similarity=True)
                    st.markdown("---")
                    
            else:
                st.error("Sorry, couldn't find recommendations for this movie.")
        else:
            st.markdown("### 👈 Select a movie to get started!")
            st.markdown("""
            This recommendation system uses **content-based filtering** to find movies similar to your selection.
            
            **How it works:**
            - Analyzes movie overviews and genres
            - Uses TF-IDF vectorization
            - Calculates cosine similarity
            - Returns most similar movies
            
            **Sample movies available:**
            - The Dark Knight
            - Inception  
            - Pulp Fiction
            - The Matrix
            - And more!
            """)
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; color: #888;">
            <p>🚀 Built with Streamlit | 🎬 Powered by Content-Based Filtering</p>
            <p>Ready to expand with TMDB 5000 dataset!</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
