import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os

def create_sequences(data, labels, window_size=60):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(labels[i+window_size])
    return np.array(X), np.array(y)

print("Loading raw data...")
try:
    df = pd.read_csv('data/raw_stock_data.csv')
    
    # Drop rows with missing values
    df = df.dropna()
    
    # Focusing on OHLCV columns (adjust column names if needed based on yfinance output)
    # yfinance multi-index columns might need handling if multiple stocks are present
    # For simplicity, if multi-level, we might need to flatten or select specific stock
    # Checking if 'Close' is in columns or if it's MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        # Flatten for simplicity if needed, or just take one stock for this demo phase
        # Assuming single stock or taking first level for now if structure allows
        pass 
    
    # For this specific project flow, let's assume we handle one stock or concatenated
    # If raw_stock_data.csv has MultiLevel columns (Price, Ticker), we need to be careful.
    # Let's inspect the first few lines to ensure we handle it right.
    # But for a standard script, let's assume a flat structure or specific columns.
    
    # Actually, yfinance download(..., group_by='ticker') might impact structure.
    # Let's try to infer or stick to the user's plan: "df[['Open', 'High', ...]]"
    # If multiple stocks, this might fail without specific selection. 
    # Let's pivot to handling just AAPL for the primary model training in this script 
    # or handle the dataframe structure dynamically.
    
    # Re-reading with header options to handle potential multi-index
    df = pd.read_csv('data/raw_stock_data.csv', header=[0, 1], index_col=0) 
    
    # Flatten columns for easier handling if it's multi-index
    # expected: (Price, Ticker) -> Price_Ticker
    # But user might want to train on one specific stock primarily?
    # User plan says: "Download data for selected stocks... and use for training"
    # Let's pick AAPL for now to ensure shape consistency as per "sequences" logic.
    
    # Let's try to just use 'Close' of AAPL if available, or just first available stock.
    # To be robust, let's just use the 'Close' of the first stock found.
    
    # Simplified approach: Use 'Close' column for label generation, OHLCV for features of ONE stock (e.g. AAPL)
    # The user plan implies training a model. Let's pick AAPL.
    
    target_stock = 'AAPL'
    
    # Extract specific stock data
    # Note: data/raw_stock_data.csv format depends on how it was saved.
    # if it was saved directly from yf.download call, it likely has MultiIndex headers.
    
    # Let's reload safely
    df_raw = pd.read_csv('data/raw_stock_data.csv', header=[0, 1], index_col=0)
    
    # Check if 'AAPL' is in the second level
    if target_stock in df_raw.columns.get_level_values(1):
        df_stock = df_raw.xs(target_stock, axis=1, level=1)
    else:
        # Fallback or maybe single level
        df_stock = pd.read_csv('data/raw_stock_data.csv', index_col=0)
    
    # Clean
    df_stock = df_stock.dropna()
    
    # Select features
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    features = df_stock[feature_cols].values
    
    # Normalize
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Create labels: 1 if next Close > current Close, else 0
    # We need to shift Close to compare
    close_prices = df_stock['Close'].values
    # y[i] is 1 if Close[i+1] > Close[i]
    # Actually, for the sequence ending at t, we want to predict t+1 movement
    # So if sequence is t-N to t, label is (Close[t+1] > Close[t])
    
    # Let's create binary labels
    # 1 if price went UP, 0 if DOWN/SAME
    labels = (np.roll(close_prices, -1) > close_prices).astype(int)
    # Last label is invalid because no next price
    labels[-1] = 0 
    
    # Create sequences
    WINDOW_SIZE = 60
    X, y = create_sequences(features_scaled, labels, WINDOW_SIZE)
    
    # Remove the last element potentially as its label might be based on rolled data (the last point)
    # Actually create_sequences loop goes up to len-window_size.
    # The label index i+window_size goes up to len.
    # We must ensure we don't include the invalid last label.
    # The last valid index for prediction is len-2 (predicting len-1 from len-2).
    # Safety: slice off last item if needed.
    X = X[:-1]
    y = y[:-1]
    
    print(f"Features shape: {X.shape}, Labels shape: {y.shape}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Save
    np.save('data/X_train.npy', X_train)
    np.save('data/X_test.npy', X_test)
    np.save('data/y_train.npy', y_train)
    np.save('data/y_test.npy', y_test)
    
    print("Preprocessing complete. Data saved to data/ directory.")

except Exception as e:
    print(f"Error during preprocessing: {e}")
