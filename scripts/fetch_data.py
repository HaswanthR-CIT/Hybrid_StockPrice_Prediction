import yfinance as yf
import pandas as pd
import os

print("Fetching data...")

try:
    # Use 'max' period to get historical and real-time data up to today
    data = yf.download('TSLA', period='max')
    if data.empty:
        print("Warning: No data fetched.")
    else:
        # yfinance returns flat columns for a single ticker.
        # We enforce a MultiIndex so preprocess_data.py parses it correctly
        if not isinstance(data.columns, pd.MultiIndex):
            data.columns = pd.MultiIndex.from_product([data.columns, ['TSLA']])
            
        print("Data fetched. Shape:", data.shape)
        output_path = 'data/raw_stock_data.csv'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        data.to_csv(output_path)
        print(f"Data saved to {output_path}")
except Exception as e:
    print(f"An error occurred: {e}")
