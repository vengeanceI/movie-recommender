"""
TMDB 5000 Dataset Setup Script
This script downloads the TMDB 5000 dataset and prepares it for the movie recommender
"""

import pandas as pd
import requests
import os
from io import StringIO

def download_tmdb_data():
    """Download TMDB 5000 datasets from GitHub"""
    
    print("📥 Downloading TMDB 5000 Movie Dataset...")
    
    # URLs for the datasets (from GitHub mirror)
    movies_url = "https://raw.githubusercontent.com/danielgrijalva/movie-recommender/master/tmdb_5000_movies.csv"
    credits_url = "https://raw.githubusercontent.com/danielgrijalva/movie-recommender/master/tmdb_5000_credits.csv"
    
    try:
        # Download movies dataset
        print("📽️ Downloading movies data...")
        movies_response = requests.get(movies_url)
        movies_response.raise_for_status()
        
        with open('tmdb_5000_movies.csv', 'w', encoding='utf-8') as f:
            f.write(movies_response.text)
        print("✅ Movies dataset downloaded successfully!")
        
        # Download credits dataset
        print("🎭 Downloading credits data...")
        credits_response = requests.get(credits_url)
        credits_response.raise_for_status()
        
        with open('tmdb_5000_credits.csv', 'w', encoding='utf-8') as f:
            f.write(credits_response.text)
        print("✅ Credits dataset downloaded successfully!")
        
        # Verify the files
        movies_df = pd.read_csv('tmdb_5000_movies.csv')
        credits_df = pd.read_csv('tmdb_5000_credits.csv')
        
        print(f"\n📊 Dataset Info:")
        print(f"Movies: {len(movies_df)} records")
        print(f"Credits: {len(credits_df)} records")
        print(f"Movies columns: {list(movies_df.columns)}")
        print(f"Credits columns: {list(credits_df.columns)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return False

def create_backup_data():
    """Create a smaller backup dataset if download fails"""
    print("🔄 Creating backup dataset...")
    
    # Sample movie data
    movies_data = {
        'id': range(1, 101),
        'title': [f'Movie {i}' for i in range(1, 101)],
        'overview': [f'This is the overview for movie {i}' for i in range(1, 101)],
        'genres': ['[{"id": 28, "name": "Action"}, {"id": 53, "name": "Thriller"}]'] * 100,
        'keywords': ['[{"id": 1, "name": "keyword1"}, {"id": 2, "name": "keyword2"}]'] * 100,
        'vote_average': [7.5] * 100,
        'vote_count': [1000] * 100,
        'popularity': [50.0] * 100,
        'release_date': ['2020-01-01'] * 100,
        'runtime': [120] * 100
    }
    
    credits_data = {
        'movie_id': range(1, 101),
        'cast': ['[{"name": "Actor 1"}, {"name": "Actor 2"}]'] * 100,
        'crew': ['[{"name": "Director 1", "job": "Director"}]'] * 100
    }
    
    movies_df = pd.DataFrame(movies_data)
    credits_df = pd.DataFrame(credits_data)
    
    movies_df.to_csv('tmdb_5000_movies.csv', index=False)
    credits_df.to_csv('tmdb_5000_credits.csv', index=False)
    
    print("✅ Backup dataset created!")

if __name__ == "__main__":
    print("🎬 TMDB 5000 Dataset Setup")
    print("=" * 30)
    
    # Check if files already exist
    if os.path.exists('tmdb_5000_movies.csv') and os.path.exists('tmdb_5000_credits.csv'):
        print("✅ Dataset files already exist!")
        
        # Check file sizes
        movies_size = os.path.getsize('tmdb_5000_movies.csv') / 1024 / 1024  # MB
        credits_size = os.path.getsize('tmdb_5000_credits.csv') / 1024 / 1024  # MB
        
        print(f"📁 Movies file size: {movies_size:.2f} MB")
        print(f"📁 Credits file size: {credits_size:.2f} MB")
        
        if movies_size < 0.1 or credits_size < 0.1:
            print("⚠️ Files seem too small, re-downloading...")
            success = download_tmdb_data()
        else:
            print("🎉 Dataset is ready!")
            success = True
    else:
        success = download_tmdb_data()
    
    if not success:
        print("⚠️ Download failed, creating backup dataset...")
        create_backup_data()
    
    print("\n🚀 Setup complete! You can now run your Streamlit app.")
    print("Command: streamlit run app.py")