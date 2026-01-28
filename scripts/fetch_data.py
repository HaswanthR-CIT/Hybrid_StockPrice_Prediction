import yfinance as yf
import pandas as pd
import os

print("Fetching data...")
stocks = ['AAPL', 'TSLA']
start_date = '2019-01-01'
end_date = '2024-01-01'

try:
    data = yf.download(stocks, start=start_date, end=end_date)
    if data.empty:
        print("Warning: No data fetched.")
    else:
        print("Data fetched. Shape:", data.shape)
        output_path = 'data/raw_stock_data.csv'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        data.to_csv(output_path)
        print(f"Data saved to {output_path}")
except Exception as e:
    print(f"An error occurred: {e}")
