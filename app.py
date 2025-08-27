import streamlit as st
import pandas as pd
import requests
import json
import gzip
import io
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page config
st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# TMDB API key from Streamlit secrets
try:
    TMDB_API_KEY = st.secrets["TMDB_API_KEY"]
except:
    TMDB_API_KEY = None
    st.warning("⚠️ TMDB API key not found in secrets. Poster images may not load.")

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
        # Load movies data from GitHub
        movies_url = f"{GITHUB_REPO_URL}{MOVIES_FILE}"
        st.info(f"📡 Loading movies from: {movies_url}")
        
        movies_response = requests.get(movies_url, timeout=30)
        movies_response.raise_for_status()
        
        movies_df = pd.read_csv(io.StringIO(movies_response.text))
        st.success(f"✅ Loaded {len(movies_df)} movies from GitHub")
        
        # Load credits data from GitHub
        credits_url = f"{GITHUB_REPO_URL}{CREDITS_FILE}"
        st.info(f"📡 Loading credits from: {credits_url}")
        
        credits_response = requests.get(credits_url, timeout=30)
        credits_response.raise_for_status()
        
        # Handle regular CSV file
        credits_df = pd.read_csv(io.StringIO(credits_response.text))
        
        st.success(f"✅ Loaded {len(credits_df)} credit records from GitHub")
        
        # Merge movies with credits
        if credits_df is not None:
            # Check column names and merge appropriately
            if 'movie_id' in credits_df.columns:
                movies_df = movies_df.merge(credits_df, left_on='id', right_on='movie_id', how='left')
            else:
                movies_df = movies_df.merge(credits_df, on='id', how='left')
            
            data_source += " (Movies + Credits)"
        else:
            data_source += " (Movies Only)"
        
        # Process the merged data
        movies_df = process_movie_data(movies_df)
        
        st.success(f"✅ Final dataset: {len(movies_df)} movies from {data_source}")
        return movies_df
        
    except requests.exceptions.RequestException as e:
        st.error(f"🌐 Network error loading from GitHub: {e}")
        return create_fallback_data()
    except Exception as e:
        st.error(f"❌ Error loading data from GitHub: {e}")
        return create_fallback_data()

@st.cache_data
def load_movie_data():
    """Main data loading function - tries GitHub first, then fallback"""
    try:
        # Try loading from GitHub
        return load_data_from_github()
    except Exception as e:
        st.warning(f"⚠️ GitHub loading failed: {e}")
        return create_fallback_data()

def process_movie_data(movies_df):
    """Process and clean movie data with credits handling"""
    try:
        # Essential columns
        essential_cols = ['id', 'title', 'overview', 'genres', 'vote_average', 'vote_count', 
                         'popularity', 'release_date', 'runtime']
        
        # Optional columns that might be present from credits
        optional_cols = ['cast', 'crew', 'keywords']
        available_cols = essential_cols + [col for col in optional_cols if col in movies_df.columns]
        
        movies_df = movies_df[available_cols].copy()
        
        # Clean basic data
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
        
        # Process keywords if available
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
        
        # Process cast data
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
        
        # Process crew data to extract director
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
        
        # Create combined features for similarity
        feature_components = [
            movies_df['overview'].fillna(''),
            movies_df['genres'].fillna(''),
        ]
        
        # Add available optional features
        if 'cast' in movies_df.columns:
            cast_clean = movies_df['cast'].fillna('').str.replace('|', ' ')
            feature_components.append(cast_clean)
        
        if 'director' in movies_df.columns:
            feature_components.append(movies_df['director'].fillna(''))
        
        if 'keywords' in movies_df.columns:
            feature_components.append(movies_df['keywords'].fillna(''))
        
        # Combine all features
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
    st.warning("⚠️ Using sample data. Please check your GitHub repository configuration.")
    
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

def display_movie_info(movie_info, width=150):
    """Display detailed movie information"""
    col_a, col_b = st.columns([1, 2])
    
    with col_a:
        poster_url = get_movie_poster(movie_info['title'], movie_info.get('id'))
        st.image(poster_url, width=width)
    
    with col_b:
        st.markdown(f"**⭐ Rating:** {movie_info['vote_average']:.1f}/10")
        if 'vote_count' in movie_info and pd.notna(movie_info['vote_count']):
            st.markdown(f"**🗳️ Votes:** {movie_info['vote_count']:,}")
        if 'release_date' in movie_info and pd.notna(movie_info['release_date']):
            year = str(movie_info['release_date'])[:4] if len(str(movie_info['release_date'])) >= 4 else 'Unknown'
            st.markdown(f"**📅 Year:** {year}")
        if 'runtime' in movie_info and pd.notna(movie_info['runtime']):
            st.markdown(f"**⏱️ Runtime:** {int(movie_info['runtime'])} min")
    
    st.markdown(f"**🎭 Genres:** {movie_info['genres']}")
    
    if 'cast' in movie_info and pd.notna(movie_info['cast']) and movie_info['cast']:
        cast_str = str(movie_info['cast'])
        if len(cast_str) > 100:
            cast_str = cast_str[:100] + "..."
        st.markdown(f"**👥 Cast:** {cast_str}")
    
    if 'director' in movie_info and pd.notna(movie_info['director']) and movie_info['director']:
        st.markdown(f"**🎬 Director:** {movie_info['director']}")

def main():
    # Header
    st.title("🎬 Movie Recommendation System")
    st.markdown("*Discover movies you'll love based on your favorites*")
    
    # Configuration section
    with st.expander("⚙️ GitHub Configuration", expanded=False):
        st.markdown("""
        **Instructions:**
        1. Repository URL: https://github.com/vengeanceI/movie-recommender
        2. Files required: tmdb_5000_movies.csv and tmdb_credits_maximum.csv
        3. App will fetch data directly from your GitHub repository
        """)
        
        st.code(f"""
        GITHUB_REPO_URL = "https://raw.githubusercontent.com/vengeanceI/movie-recommender/main/"
        MOVIES_FILE = "{MOVIES_FILE}"
        CREDITS_FILE = "{CREDITS_FILE}"
        """)
    
    st.markdown("---")
    
    # Load data
    with st.spinner("🔡 Loading movie database from GitHub..."):
        movies_df = load_movie_data()
        
        if movies_df is None or len(movies_df) == 0:
            st.error("❌ Could not load movie data. Please check your GitHub repository configuration.")
            return
        
        # Create similarity matrix
        with st.spinner("🧮 Computing movie similarities..."):
            similarity_matrix = create_similarity_matrix(movies_df)
            
            if similarity_matrix is None:
                st.error("❌ Could not create recommendation engine.")
                return
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Controls")
        n_recommendations = st.slider("Number of recommendations", 1, 20, 6)
        
        st.markdown("---")
        st.markdown("### 📊 Dataset Info")
        st.info(f"🎬 Movies: {len(movies_df):,}")
        st.info(f"⭐ Avg rating: {movies_df['vote_average'].mean():.1f}/10")
        
        # Feature availability
        st.markdown("### 📋 Available Features")
        st.write("✅ Movie overviews & genres")
        st.write("✅ Ratings & popularity")
        
        if 'cast' in movies_df.columns and movies_df['cast'].notna().any():
            cast_coverage = (movies_df['cast'] != '').sum() / len(movies_df) * 100
            st.write(f"✅ Cast info ({cast_coverage:.0f}% coverage)")
        
        if 'director' in movies_df.columns and movies_df['director'].notna().any():
            director_coverage = (movies_df['director'] != '').sum() / len(movies_df) * 100
            st.write(f"✅ Director info ({director_coverage:.0f}% coverage)")
        
        # Top rated movies
        st.markdown("### 🏆 Top Rated")
        top_movies = movies_df.nlargest(5, 'vote_average')[['title', 'vote_average']]
        for _, movie in top_movies.iterrows():
            st.write(f"⭐ {movie['vote_average']:.1f} - {movie['title']}")
    
    # Main content
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("🔍 Select a Movie")
        
        # Search functionality
        search_query = st.text_input(
            "🔎 Search movies:",
            placeholder="Type movie name...",
            help="Search for any movie in our database to get recommendations"
        )
        
        if search_query:
            filtered_movies = movies_df[
                movies_df['title'].str.lower().str.contains(search_query.lower(), na=False)
            ]['title'].tolist()[:25]
        else:
            # Show popular/highly rated movies by default
            popular_movies = movies_df.nlargest(40, 'popularity')
            top_rated = movies_df.nlargest(40, 'vote_average')
            combined = pd.concat([popular_movies, top_rated]).drop_duplicates('title')
            filtered_movies = combined['title'].tolist()
        
        if filtered_movies:
            selected_movie = st.selectbox("Choose a movie:", filtered_movies)
        else:
            st.warning("No movies found matching your search.")
            selected_movie = None
        
        # Display selected movie info
        if selected_movie:
            movie_info = movies_df[movies_df['title'] == selected_movie].iloc[0]
            
            st.markdown("### 📋 Movie Details")
            display_movie_info(movie_info, width=140)
            
            # Plot summary
            with st.expander("📖 Full Plot Summary"):
                st.write(movie_info['overview'])
    
    with col2:
        st.header("🎯 Recommended Movies")
        
        if selected_movie:
            with st.spinner("🤖 Finding similar movies..."):
                recommendations = recommend_movies(
                    selected_movie, movies_df, similarity_matrix, n_recommendations
                )
                
                if len(recommendations) > 0:
                    st.success(f"Found {len(recommendations)} great recommendations!")
                    
                    for idx, (_, movie) in enumerate(recommendations.iterrows()):
                        with st.container():
                            st.markdown(f"### {idx+1}. {movie['title']}")
                            
                            col_x, col_y, col_z = st.columns([1, 2, 1])
                            
                            with col_x:
                                poster_url = get_movie_poster(movie['title'], movie.get('id'))
                                st.image(poster_url, width=90)
                            
                            with col_y:
                                st.markdown(f"**⭐ Rating:** {movie['vote_average']:.1f}/10")
                                
                                if 'vote_count' in movie and pd.notna(movie['vote_count']):
                                    st.markdown(f"**🗳️ Votes:** {movie['vote_count']:,}")
                                
                                st.markdown(f"**🎭 Genres:** {movie['genres']}")
                                
                                if 'cast' in movie and pd.notna(movie['cast']) and movie['cast']:
                                    cast_display = str(movie['cast'])[:60] + "..." if len(str(movie['cast'])) > 60 else str(movie['cast'])
                                    st.markdown(f"**👥 Cast:** {cast_display}")
                                
                                with st.expander(f"About {movie['title']}"):
                                    st.write(movie['overview'])
                            
                            with col_z:
                                similarity_score = movie.get('similarity_score', 0) * 100
                                st.metric("Match", f"{similarity_score:.0f}%")
                            
                            st.markdown("---")
                
                else:
                    st.warning("😞 No recommendations found for this movie.")
        else:
            st.info("👆 Search and select a movie above to get personalized recommendations!")
            
            # Show some featured recommendations
            st.markdown("### 🌟 Featured Movies")
            featured = movies_df.nlargest(3, 'vote_average')
            
            for _, movie in featured.iterrows():
                with st.container():
                    col_feat1, col_feat2 = st.columns([1, 3])
                    
                    with col_feat1:
                        poster_url = get_movie_poster(movie['title'], movie.get('id'))
                        st.image(poster_url, width=80)
                    
                    with col_feat2:
                        st.markdown(f"**{movie['title']}**")
                        st.markdown(f"⭐ {movie['vote_average']:.1f}/10 • {movie['genres']}")
                        if len(movie['overview']) > 120:
                            st.markdown(f"{movie['overview'][:120]}...")
                        else:
                            st.markdown(movie['overview'])
                    
                    st.markdown("---")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 0.8em;'>
            🎬 Data from TMDB via GitHub Repository | Built with ❤️ using Streamlit<br>
            💡 Content-based filtering using movie overviews, genres, cast & crew
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
