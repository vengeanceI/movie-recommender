import streamlit as st
import pandas as pd
import requests
import json

# Page config
st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# Sample movie database (fallback)
SAMPLE_MOVIES = [
    {"id": 1, "title": "The Shawshank Redemption", "genres": "Drama", "vote_average": 9.3, "overview": "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency."},
    {"id": 2, "title": "The Godfather", "genres": "Crime Drama", "vote_average": 9.2, "overview": "The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son."},
    {"id": 3, "title": "The Dark Knight", "genres": "Action Superhero", "vote_average": 9.0, "overview": "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests."},
    {"id": 4, "title": "Pulp Fiction", "genres": "Crime Thriller", "vote_average": 8.9, "overview": "The lives of two mob hitmen, a boxer, a gangster and his wife intertwine in four tales of violence and redemption."},
    {"id": 5, "title": "Forrest Gump", "genres": "Drama Comedy", "vote_average": 8.8, "overview": "The presidencies of Kennedy and Johnson through the eyes of an Alabama man with an IQ of 75."},
    {"id": 6, "title": "Inception", "genres": "Sci-Fi Thriller", "vote_average": 8.8, "overview": "A thief who steals corporate secrets through dream-sharing technology is given the inverse task of planting an idea."},
    {"id": 7, "title": "The Matrix", "genres": "Sci-Fi Action", "vote_average": 8.7, "overview": "A hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers."},
    {"id": 8, "title": "Goodfellas", "genres": "Crime Drama", "vote_average": 8.7, "overview": "The story of Henry Hill and his life in the mob, covering his relationship with his wife Karen Hill."},
    {"id": 9, "title": "The Silence of the Lambs", "genres": "Thriller Horror", "vote_average": 8.6, "overview": "A young FBI cadet must receive the help of an incarcerated and manipulative cannibal killer to help catch another serial killer."},
    {"id": 10, "title": "Saving Private Ryan", "genres": "War Drama", "vote_average": 8.6, "overview": "Following the Normandy Landings, a group of U.S. soldiers go behind enemy lines to retrieve a paratrooper."},
    {"id": 11, "title": "Interstellar", "genres": "Sci-Fi Drama", "vote_average": 8.6, "overview": "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival."},
    {"id": 12, "title": "The Avengers", "genres": "Action Adventure", "vote_average": 8.0, "overview": "Earth's mightiest heroes must come together and learn to fight as a team to stop the mischievous Loki."},
    {"id": 13, "title": "Iron Man", "genres": "Action Superhero", "vote_average": 7.9, "overview": "After being held captive in an Afghan cave, billionaire engineer Tony Stark creates a unique weaponized suit of armor."},
    {"id": 14, "title": "Spider-Man", "genres": "Action Superhero", "vote_average": 7.3, "overview": "After being bitten by a genetically altered spider, nerdy high school student Peter Parker is endowed with amazing powers."},
    {"id": 15, "title": "Batman Begins", "genres": "Action Superhero", "vote_average": 8.2, "overview": "After training with his mentor, Batman begins his fight to free crime-ridden Gotham City from corruption."},
    {"id": 16, "title": "Fight Club", "genres": "Drama Thriller", "vote_average": 8.8, "overview": "An insomniac office worker and a devil-may-care soapmaker form an underground fight club."},
    {"id": 17, "title": "Titanic", "genres": "Romance Drama", "vote_average": 7.8, "overview": "A seventeen-year-old aristocrat falls in love with a kind but poor artist aboard the luxurious, ill-fated R.M.S. Titanic."},
    {"id": 18, "title": "Avatar", "genres": "Sci-Fi Adventure", "vote_average": 7.8, "overview": "A paraplegic Marine dispatched to the moon Pandora on a unique mission becomes torn between following orders and protecting an alien civilization."},
    {"id": 19, "title": "Jurassic Park", "genres": "Adventure Sci-Fi", "vote_average": 8.1, "overview": "A pragmatic paleontologist visiting an almost complete theme park is tasked with protecting a couple of kids after a power failure causes the park's cloned dinosaurs to run loose."},
    {"id": 20, "title": "Terminator 2", "genres": "Sci-Fi Action", "vote_average": 8.5, "overview": "A cyborg, identical to the one who failed to kill Sarah Connor, must now protect her teenage son John Connor from a more advanced and powerful cyborg."}
]

@st.cache_data
def load_movie_data():
    """Load movie data - tries CSV first, then uses sample data"""
    try:
        # Try to load TMDB dataset
        movies = pd.read_csv('tmdb_5000_movies.csv')
        credits = pd.read_csv('tmdb_5000_credits.csv')
        
        # Basic processing
        movies = movies.merge(credits, left_on='id', right_on='movie_id', how='left')
        movies = movies[['id', 'title', 'overview', 'genres', 'vote_average', 'vote_count', 'popularity', 'release_date']]
        movies = movies.dropna(subset=['title', 'overview'])
        
        # Simple genre extraction
        def extract_genre_names(text):
            try:
                data = json.loads(text.replace("'", '"'))
                names = [item['name'] for item in data[:2]]
                return ' '.join(names)
            except:
                return 'Unknown'
        
        movies['genres'] = movies['genres'].fillna('[]').apply(extract_genre_names)
        st.success(f"✅ Loaded {len(movies)} movies from TMDB dataset!")
        return movies
        
    except Exception as e:
        # Fallback to sample data
        st.warning("⚠️ TMDB dataset not found, using sample data.")
        st.info("💡 To use full dataset, upload 'tmdb_5000_movies.csv' and 'tmdb_5000_credits.csv'")
        return pd.DataFrame(SAMPLE_MOVIES)

def simple_recommend(selected_movie, movies_df, n_recs=5):
    """Simple recommendation based on genre similarity"""
    try:
        selected_row = movies_df[movies_df['title'] == selected_movie].iloc[0]
        selected_genres = selected_row['genres'].lower()
        
        # Calculate simple similarity based on genres
        similarities = []
        for idx, row in movies_df.iterrows():
            if row['title'] == selected_movie:
                continue
            
            movie_genres = row['genres'].lower()
            
            # Simple word overlap similarity
            selected_words = set(selected_genres.split())
            movie_words = set(movie_genres.split())
            
            if selected_words and movie_words:
                similarity = len(selected_words.intersection(movie_words)) / len(selected_words.union(movie_words))
            else:
                similarity = 0
            
            similarities.append((idx, similarity, row['vote_average']))
        
        # Sort by similarity and rating
        similarities.sort(key=lambda x: (x[1], x[2]), reverse=True)
        
        # Get top recommendations
        top_indices = [sim[0] for sim in similarities[:n_recs]]
        recommendations = movies_df.iloc[top_indices].copy()
        
        return recommendations
        
    except Exception as e:
        st.error(f"Error in recommendation: {e}")
        return pd.DataFrame()

def get_movie_poster(movie_title):
    """Get movie poster from TMDB API"""
    try:
        api_key = "65fc826f62592ec1235e593cf3479495"
        url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={movie_title}"
        response = requests.get(url, timeout=3)
        data = response.json()
        
        if 'results' in data and data['results'] and data['results'][0].get('poster_path'):
            poster_path = data['results'][0]['poster_path']
            return f"https://image.tmdb.org/t/p/w300{poster_path}"
    except:
        pass
    
    return "https://via.placeholder.com/200x300/2e2e2e/ffffff?text=🎬"

def main():
    # Header
    st.title("🎬 Movie Recommendation System")
    st.markdown("*Simple & Fast Movie Recommendations*")
    st.markdown("---")
    
    # Load data
    with st.spinner("🔄 Loading movie database..."):
        movies_df = load_movie_data()
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Settings")
        n_recs = st.slider("Number of recommendations", 1, 10, 5)
        
        st.markdown("---")
        st.markdown("### 📊 Stats")
        st.info(f"📽️ Movies: {len(movies_df)}")
        st.info(f"⭐ Avg Rating: {movies_df['vote_average'].mean():.1f}/10")
        
        st.markdown("### 🏆 Top Rated")
        top_movies = movies_df.nlargest(5, 'vote_average')[['title', 'vote_average']]
        for _, movie in top_movies.iterrows():
            st.write(f"⭐ {movie['vote_average']:.1f} - {movie['title']}")
    
    # Main interface
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("🔍 Select Movie")
        
        # Search
        search = st.text_input("🔎 Search movies:", placeholder="Type movie name...")
        
        if search:
            filtered = movies_df[movies_df['title'].str.contains(search, case=False, na=False)]
            options = filtered['title'].tolist()[:15]
        else:
            # Show top rated movies
            top_rated = movies_df.nlargest(20, 'vote_average')
            options = top_rated['title'].tolist()
        
        if options:
            selected = st.selectbox("Choose a movie:", options)
        else:
            st.warning("No movies found!")
            selected = None
        
        if selected:
            movie_info = movies_df[movies_df['title'] == selected].iloc[0]
            
            st.markdown("### 📋 Details")
            
            # Show poster and details
            col_a, col_b = st.columns([1, 2])
            with col_a:
                poster = get_movie_poster(selected)
                st.image(poster, width=120)
            
            with col_b:
                st.markdown(f"**⭐ Rating:** {movie_info['vote_average']:.1f}/10")
                if 'vote_count' in movie_info:
                    st.markdown(f"**🗳️ Votes:** {movie_info['vote_count']:,}")
                st.markdown(f"**🎭 Genre:** {movie_info['genres']}")
            
            with st.expander("📖 Plot"):
                st.write(movie_info['overview'])
    
    with col2:
        st.header("🎯 Recommendations")
        
        if selected:
            with st.spinner("🤖 Finding similar movies..."):
                recs = simple_recommend(selected, movies_df, n_recs)
                
                if len(recs) > 0:
                    st.success(f"Found {len(recs)} recommendations!")
                    
                    for idx, (_, movie) in enumerate(recs.iterrows()):
                        with st.container():
                            col_x, col_y = st.columns([1, 3])
                            
                            with col_x:
                                poster = get_movie_poster(movie['title'])
                                st.image(poster, width=100)
                            
                            with col_y:
                                st.markdown(f"### {idx+1}. {movie['title']}")
                                st.markdown(f"**⭐ Rating:** {movie['vote_average']:.1f}/10")
                                st.markdown(f"**🎭 Genre:** {movie['genres']}")
                                
                                with st.expander(f"More about {movie['title']}"):
                                    st.write(movie['overview'])
                            
                            st.markdown("---")
                else:
                    st.warning("😔 No recommendations found.")
        else:
            st.info("👆 Select a movie to get recommendations!")
    
    # Footer
    st.markdown("---")
    st.markdown("<div style='text-align: center; color: #666;'>🎬 Made with ❤️ using Streamlit</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
