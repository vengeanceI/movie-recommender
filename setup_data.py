"""
TMDB 5000 Dataset Setup Script
Downloads the TMDB 5000 dataset if not found.
"""

import pandas as pd
import requests
import os

def download_tmdb_data():
    """Download TMDB 5000 datasets from GitHub mirror"""
    movies_url = "https://raw.githubusercontent.com/danielgrijalva/movie-recommender/master/tmdb_5000_movies.csv"
    credits_url = "https://raw.githubusercontent.com/danielgrijalva/movie-recommender/master/tmdb_5000_credits.csv"
    
    try:
        # Download movies dataset
        movies_response = requests.get(movies_url, timeout=15)
        movies_response.raise_for_status()
        with open('tmdb_5000_movies.csv', 'w', encoding='utf-8') as f:
            f.write(movies_response.text)
        
        # Download credits dataset
        credits_response = requests.get(credits_url, timeout=15)
        credits_response.raise_for_status()
        with open('tmdb_5000_credits.csv', 'w', encoding='utf-8') as f:
            f.write(credits_response.text)
        
        return True
    except Exception as e:
        print("❌ Error downloading dataset:", e)
        return False
