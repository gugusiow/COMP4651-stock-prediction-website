# data-pipeline/src/data_collector.py
import yfinance as yf
import pandas as pd
import os
import sys
import time

# Import configuration
sys.path.append(os.path.dirname(__file__))
from config import *

class DataCollector:
    def __init__(self):
        self.stocks = STOCKS
        
    def fetch_single_stock(self, ticker):
        """Fetch data for a single stock using Ticker object"""
        try:
            print(f"Fetching {ticker}...")
            
            # Use Ticker object instead of download for single stock
            stock = yf.Ticker(ticker)
            data = stock.history(period=DATA_PERIOD, interval=DATA_INTERVAL, auto_adjust=True)
            
            if data.empty:
                print(f"No data for {ticker}")
                return None
            
            print(f"{ticker} columns: {data.columns.tolist()}")
            print(f"{ticker} shape: {data.shape}")
            
            # Keep only needed columns
            data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
            data = data.dropna()
            
            print(f"{ticker}: {len(data)} records")
            return data
            
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            return None
    
    def fetch_all_stocks(self):
        """Fetch data for all stocks"""
        print("Starting data collection...")
        print(f"Total stocks: {len(self.stocks)}")
        print(f"Interval: {DATA_INTERVAL}, Period: {DATA_PERIOD}")
        
        all_data = {}
        successful = 0
        
        for ticker in self.stocks:
            data = self.fetch_single_stock(ticker)
            if data is not None:
                all_data[ticker] = data
                successful += 1
            time.sleep(1)  # Avoid rate limiting
        
        print(f"Data collection completed: {successful}/{len(self.stocks)} successful")
        return all_data
    
    def save_data(self, all_data):
        """Save data to CSV files"""
        print("Saving data...")
        
        for ticker, data in all_data.items():
            filepath = os.path.join(RAW_DATA_DIR, f"{ticker}.csv")
            
            # Reset index to convert datetime index to column
            data_with_dates = data.reset_index()
            
            # Rename the first column to 'Datetime'
            data_with_dates = data_with_dates.rename(columns={'Date': 'Datetime'})
            
            # Save to CSV
            data_with_dates.to_csv(filepath, index=False)
            
            print(f"Saved {ticker}: {len(data_with_dates)} records")
            print(f"  Sample Close: {data_with_dates['Close'].head(3).tolist()}")
        
        print("All data saved successfully!")

def main():
    collector = DataCollector()
    all_data = collector.fetch_all_stocks()
    
    if all_data:
        collector.save_data(all_data)
        return True
    return False

if __name__ == "__main__":
    main()