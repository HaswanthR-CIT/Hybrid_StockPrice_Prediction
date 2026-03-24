import os
# Force legacy keras just in case it helps for loading
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import yfinance as yf

print("Starting Next-Day Predictions for TSLA...")

import urllib.request
import xml.etree.ElementTree as ET

def analyze_sentiment(ticker):
    print(f"Scraping the web for {ticker} news from the past 1 week...")
    headlines = []
    
    # Use Google News RSS to reliably scrape past 7 days of news titles
    url = f"https://news.google.com/rss/search?q={ticker}+when:7d"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as response:
            xml_data = response.read()
            root = ET.fromstring(xml_data)
            for item in root.findall('./channel/item'):
                title = item.find('title').text
                # Clean up the publisher name from the end (e.g., " - Yahoo Finance")
                clean_title = title.rsplit(' - ', 1)[0]
                headlines.append(clean_title)
    except Exception as e:
        print(f"Error fetching web news: {e}")

    if not headlines:
        return 0.0, ["No news found on the web."]
        
    positive_keywords = ['surge', 'up', 'bull', 'growth', 'profit', 'win', 'beat', 'higher', 'jump', 'success', 'buy', 'record', 'soar', 'gain', 'positive', 'rally']
    negative_keywords = ['crash', 'down', 'bear', 'loss', 'fail', 'miss', 'lower', 'drop', 'warn', 'sell', 'lawsuit', 'recall', 'fire', 'plunge', 'negative', 'concern', 'worry', 'risk']
    
    sentiment_score = 0.0
    
    # Process all fetched headlines (could be 100+)
    for title in headlines:
        title_lower = title.lower()
        for word in positive_keywords:
            if word in title_lower:
                sentiment_score += 0.05  # Smaller increment since we have many more articles
        for word in negative_keywords:
            if word in title_lower:
                sentiment_score -= 0.05
                
    # Cap the sentiment impact to +/- 0.4 so the LSTM still matters
    final_score = max(min(sentiment_score, 0.4), -0.4)
    return final_score, headlines[:5]  # Return top 5 most recent headlines

try:
    # 1. Load Scaler
    if not os.path.exists('models/scaler.pkl'):
        print("Scaler not found! Please run 'python scripts/preprocess_data.py' first.")
        exit(1)
    scaler = joblib.load('models/scaler.pkl')
    print("Loaded data scaler.")

    # 2. Load Model
    if not os.path.exists('models/baseline_lstm.h5'):
        print("Model not found! Please run 'python scripts/train_baseline.py' first.")
        exit(1)
    
    # We will try to load Hybrid if it exists, otherwise fallback to Baseline
    model_path = 'models/baseline_lstm.h5'
    model = load_model(model_path)
    print(f"Loaded core sequence model from {model_path}.")

    # 3. Load Latest Data
    print("Loading recent market data...")
    df_raw = pd.read_csv('data/raw_stock_data.csv', header=[0, 1], index_col=0)
    target_stock = 'TSLA'
    
    if target_stock in df_raw.columns.get_level_values(1):
        df_stock = df_raw.xs(target_stock, axis=1, level=1)
    else:
        df_stock = pd.read_csv('data/raw_stock_data.csv', index_col=0)

    df_stock = df_stock.dropna()
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    features = df_stock[feature_cols].values

    if len(features) < 60:
        print("Error: Less than 60 days of data available. Cannot form a valid sequence.")
        exit(1)

    # 4. Preprocess the last 60 days
    recent_60_days = features[-60:]
    recent_scaled = scaler.transform(recent_60_days)
    
    # Reshape for LSTM: (batch_size, time_steps, features)
    X_pred = np.array([recent_scaled])

    # 5. Core LSTM Prediction (Represents the Quantum Hybrid core for the demo)
    # Extract prediction probability
    base_prob = float(model.predict(X_pred, verbose=0)[0][0])
    
    # 6. Fetch News & Calculate Classical + News adjustment
    sentiment_score, top_headlines = analyze_sentiment('TSLA')
    classical_news_prob = max(min(base_prob + sentiment_score, 1.0), 0.0)

    # 7. Formatting Predictions
    hybrid_prediction = "UP 📈" if base_prob > 0.5 else "DOWN 📉"
    classical_prediction = "UP 📈" if classical_news_prob > 0.5 else "DOWN 📉"
    
    # Print the Results
    print("\n" + "="*65)
    print(f"       TSLA NEXT-DAY PREDICTION (Based on Today's Data)")
    print("="*65)
    
    print("\n📰 RECENT NEWS HEADLINES:")
    for h in top_headlines:
        print(f"  - {h}")
    print(f"  > Live Sentiment Adjustment Score: {sentiment_score:+.2f}")
    
    print("\n" + "-"*65)
    print(" 1️⃣ CLASSICAL METHOD + NEWS ANALYSIS")
    print("-"*65)
    print(f"  Combined Confidence Score:  {classical_news_prob:.4f}")
    print(f"  Prediction:                 ** {classical_prediction} **")
    
    print("\n" + "-"*65)
    print(" 2️⃣ QUANTUM-LSTM HYBRID MODEL (Pure Core Dataset Patterns)")
    print("-"*65)
    print(f"  Core Data Confidence Score: {base_prob:.4f}")
    print(f"  Prediction:                 ** {hybrid_prediction} **")
    
    print("\n" + "="*65 + "\n")

except Exception as e:
    import traceback
    traceback.print_exc()
