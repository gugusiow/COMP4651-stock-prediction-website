# data-pipeline/src/config.py
import os

# Get absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")

# 7 Stocks for training
STOCKS = ['AAPL', 'MSFT', 'TSLA', 'JPM', 'AMZN', 'XOM', 'JNJ']
# STOCKS = ['AAPL'] # single stock for testing

# Data parameters - Daily data for 5 years
DATA_INTERVAL = "1d"    # Daily data
DATA_PERIOD = "5y"      # 5 years
SEQUENCE_LENGTH = 60    # 60 days (approx 3 months)
TEST_SIZE = 0.2
CLASSIFICATION_THRESHOLD = 0.001  # 0.1% price movement

# File paths
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Create directories
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

print("Configuration loaded: 7 stocks, daily data over 5 years")
print(f"Stocks: {STOCKS}")
print(f"Sequence length: {SEQUENCE_LENGTH} days")