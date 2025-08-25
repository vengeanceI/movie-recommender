import streamlit as st
import pandas as pd
import requests
import json
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page config
st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# Initialize session state
if 'n_recs' not in st.session_state:
    st.session_state.n_recs = 5
if 'movies_df' not in st.session_state:
    st.session_state.movies_df = None
if 'similarity_matrix' not in st.session_state:
    st.session_state.similarity_matrix = None

@st.cache_data
def load_tmdb_data():
    """Load and process TMDB 5000 dataset"""
    try:
        # Load the datasets
        movies = pd.read_csv('tmdb_5000_movies.csv')
        credits = pd.read_csv('tmdb_5000_credits.csv')
        
        # Merge datasets on title and id
        movies = movies.merge(credits, left_on='id', right_on='movie_id', how='left')
        
        # Keep only necessary columns
        movies = movies[['id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew', 
                        'vote_average', 'vote_count', 'popularity', 'release_date', 'runtime']]
        
        # Clean the data
        movies = movies.dropna(subset=['title', 'overview'])
        
        # Process JSON columns
        def extract_names(text, count=3):
            try:
                data = json.loads(text.replace("'", '"'))
                names = [item['name'] for item in data[:count]]
                return ' '.join(names)
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
        
        # Extract features
        movies['genres'] = movies['genres'].fillna('[]').apply(lambda x: extract_names(x, 3))
        movies['keywords'] = movies['keywords'].fillna('[]').apply(lambda x: extract_names(x, 5))
        movies['cast'] = movies['cast'].fillna('[]').apply(lambda x: extract_names(x, 3))
        movies['director'] = movies['crew'].fillna('[]').apply(extract_director)
        
        # Create combined features for recommendation
        movies['combined_features'] = (
            movies['overview'].fillna('') + ' ' + 
            movies['genres'].fillna('') + ' ' + 
            movies['keywords'].fillna('') + ' ' + 
            movies['cast'].fillna('') + ' ' + 
            movies['director'].fillna('')
        )
        
        # Remove movies with very short descriptions
        movies = movies[movies['combined_features'].str.len() > 20]
        
        # Reset index
        movies = movies.reset_index(drop=True)
        
        return movies
        
    except Exception as e:
        st.error(f"Error loading TMDB data: {e}")
        return None

def simple_similarity(text1, text2):
    """Simple similarity calculation without sklearn"""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    return len(intersection) / len(union) if union else 0

@st.cache_data
def create_similarity_matrix(movies_df):
    """Create similarity matrix for recommendations"""
    try:
        if SKLEARN_AVAILABLE:
            # Use CountVectorizer for text similarity
            cv = CountVectorizer(max_features=5000, stop_words='english', lowercase=True)
            vectors = cv.fit_transform(movies_df['combined_features']).toarray()
            similarity = cosine_similarity(vectors)
            return similarity
        else:
            # Fallback: Simple similarity calculation
            st.info("Using simple similarity calculation...")
            n_movies = len(movies_df)
            similarity = [[0.0] * n_movies for _ in range(n_movies)]
            
            features = movies_df['combined_features'].tolist()
            
            for i in range(n_movies):
                for j in range(n_movies):
                    if i != j:
                        similarity[i][j] = simple_similarity(features[i], features[j])
                    else:
                        similarity[i][j] = 1.0
            
            return similarity
            
    except Exception as e:
        st.error(f"Error creating similarity matrix: {e}")
        return None

def get_movie_poster(movie_title, tmdb_id=None):
    """Get movie poster from TMDB API"""
    try:
        # TMDB API key - you can get this free from https://www.themoviedb.org/settings/api
        api_key = "8265bd1679663a7ea12ac168da84d2e8"  # Replace with your API key
        
        if tmdb_id:
            url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={api_key}"
        else:
            url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={movie_title}"
        
        response = requests.get(url, timeout=5)
        data = response.json()
        
        if tmdb_id and 'poster_path' in data and data['poster_path']:
            return f"https://image.tmdb.org/t/p/w300{data['poster_path']}"
        elif 'results' in data and data['results'] and data['results'][0]['poster_path']:
            poster_path = data['results'][0]['poster_path']
            return f"https://image.tmdb.org/t/p/w300{poster_path}"
    except:
        pass
    
    return "https://via.placeholder.com/300x450/1f1f1f/ffffff?text=No+Poster"

def recommend_movies(movie_title, movies_df, similarity_matrix, n_recommendations=5):
    """Get movie recommendations based on similarity"""
    try:
        # Find the movie index
        movie_indices = movies_df[movies_df['title'].str.lower().str.contains(movie_title.lower(), na=False)].index
        
        if len(movie_indices) == 0:
            return pd.DataFrame()
        
        movie_idx = movie_indices[0]
        
        # Get similarity scores
        sim_scores = list(enumerate(similarity_matrix[movie_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top recommendations (excluding the movie itself)
        sim_scores = sim_scores[1:n_recommendations+1]
        
        movie_indices = [i[0] for i in sim_scores]
        recommended_movies = movies_df.iloc[movie_indices].copy()
        recommended_movies['similarity_score'] = [score[1] for score in sim_scores]
        
        return recommended_movies
    
    except Exception as e:
        st.error(f"Error in recommendation: {e}")
        return pd.DataFrame()

def main():
    # Header
    st.title("🎬 TMDB Movie Recommendation System")
    st.markdown("*Powered by TMDB 5000 Dataset*")
    st.markdown("---")
    
    # Load data
    with st.spinner("🔄 Loading TMDB 5000 movie database..."):
        if st.session_state.movies_df is None:
            movies_df = load_tmdb_data()
            if movies_df is not None:
                st.session_state.movies_df = movies_df
                with st.spinner("🧮 Computing movie similarities..."):
                    similarity_matrix = create_similarity_matrix(movies_df)
                    st.session_state.similarity_matrix = similarity_matrix
            else:
                st.error("❌ Failed to load TMDB dataset. Please ensure the CSV files are uploaded.")
                st.info("📋 Required files: `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv`")
                return
    
    movies_df = st.session_state.movies_df
    similarity_matrix = st.session_state.similarity_matrix
    
    if movies_df is None or similarity_matrix is None:
        st.error("❌ Data loading failed. Please refresh and try again.")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Settings")
        st.session_state.n_recs = st.slider(
            "Number of recommendations",
            min_value=1,
            max_value=20,
            value=st.session_state.n_recs
        )
        
        st.markdown("---")
        st.markdown("### 📊 Dataset Stats")
        st.info(f"📽️ Total movies: {len(movies_df):,}")
        st.info(f"⭐ Avg rating: {movies_df['vote_average'].mean():.1f}/10")
        st.info(f"📅 Year range: {movies_df['release_date'].str[:4].min()} - {movies_df['release_date'].str[:4].max()}")
        
        # Top rated movies
        st.markdown("### 🏆 Top Rated Movies")
        top_movies = movies_df.nlargest(5, 'vote_average')[['title', 'vote_average']]
        for _, movie in top_movies.iterrows():
            st.write(f"⭐ {movie['vote_average']:.1f} - {movie['title']}")
    
    # Main interface
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("🔍 Select a Movie")
        
        # Search functionality
        search_query = st.text_input("🔎 Search for a movie:", placeholder="Type movie name...")
        
        if search_query:
            # Filter movies based on search
            filtered_movies = movies_df[
                movies_df['title'].str.lower().str.contains(search_query.lower(), na=False)
            ]['title'].tolist()[:20]  # Limit to 20 results
        else:
            # Show popular movies
            filtered_movies = movies_df.nlargest(50, 'popularity')['title'].tolist()
        
        if filtered_movies:
            selected_movie = st.selectbox(
                "Choose a movie:",
                options=filtered_movies,
                key="movie_selector"
            )
        else:
            st.warning("No movies found matching your search.")
            selected_movie = None
        
        if selected_movie:
            # Display selected movie info
            movie_info = movies_df[movies_df['title'] == selected_movie].iloc[0]
            
            st.markdown("### 📋 Movie Details")
            
            col_a, col_b = st.columns([1, 2])
            with col_a:
                poster_url = get_movie_poster(selected_movie, movie_info.get('id'))
                st.image(poster_url, width=150)
            
            with col_b:
                st.markdown(f"**⭐ Rating:** {movie_info['vote_average']:.1f}/10")
                st.markdown(f"**🗳️ Votes:** {movie_info['vote_count']:,}")
                st.markdown(f"**📅 Release:** {movie_info['release_date'][:4] if pd.notna(movie_info['release_date']) else 'Unknown'}")
                if pd.notna(movie_info['runtime']):
                    st.markdown(f"**⏱️ Runtime:** {int(movie_info['runtime'])} min")
            
            st.markdown(f"**🎭 Genres:** {movie_info['genres']}")
            st.markdown(f"**👥 Cast:** {movie_info['cast']}")
            st.markdown(f"**🎬 Director:** {movie_info['director']}")
            
            with st.expander("📝 Plot Summary"):
                st.write(movie_info['overview'])
    
    with col2:
        st.header("🎯 Recommended Movies")
        
        if selected_movie:
            with st.spinner("🤖 Finding similar movies..."):
                recommendations = recommend_movies(
                    selected_movie, 
                    movies_df, 
                    similarity_matrix, 
                    st.session_state.n_recs
                )
                
                if len(recommendations) > 0:
                    st.success(f"Found {len(recommendations)} recommendations!")
                    
                    # Display recommendations
                    for idx, (_, movie) in enumerate(recommendations.iterrows()):
                        with st.container():
                            col_x, col_y, col_z = st.columns([1, 2, 1])
                            
                            with col_x:
                                poster_url = get_movie_poster(movie['title'], movie.get('id'))
                                st.image(poster_url, width=100)
                            
                            with col_y:
                                st.markdown(f"### {idx+1}. {movie['title']}")
                                st.markdown(f"**⭐ Rating:** {movie['vote_average']:.1f}/10 ({movie['vote_count']:,} votes)")
                                st.markdown(f"**📅 Year:** {movie['release_date'][:4] if pd.notna(movie['release_date']) else 'Unknown'}")
                                st.markdown(f"**🎭 Genres:** {movie['genres']}")
                                
                                with st.expander(f"Read more about {movie['title']}"):
                                    st.write(movie['overview'])
                            
                            with col_z:
                                similarity_percentage = movie['similarity_score'] * 100
                                st.metric("Match", f"{similarity_percentage:.0f}%")
                            
                            st.markdown("---")
                
                else:
                    st.warning("😔 No recommendations found for this movie.")
        else:
            st.info("👆 Please search and select a movie to get recommendations!")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            🎬 Movie data powered by <a href='https://www.themoviedb.org/' target='_blank'>TMDB</a> | 
            Made with ❤️ using Streamlit
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
