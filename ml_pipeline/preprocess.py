# data-pipeline/src/preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import sys
import json

# Import configuration
sys.path.append(os.path.dirname(__file__))
from config import *

class DataPreprocessor:
    def __init__(self):
        self.scalers = {}
        
    def load_data(self):
        """Load all stock data"""
        print("Loading data...")
        all_data = {}
        
        for ticker in STOCKS:
            filepath = os.path.join(RAW_DATA_DIR, f"{ticker}.csv")
            if os.path.exists(filepath):
                try:
                    # Load the data
                    data = pd.read_csv(filepath)
                    
                    # Set Datetime as index
                    data = data.set_index('Datetime')
                    data.index = pd.to_datetime(data.index)
                    
                    # Ensure all columns are numeric
                    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                        data[col] = pd.to_numeric(data[col], errors='coerce')
                    
                    # Remove any NaN values
                    data = data.dropna()
                    
                    print(f"Loaded {ticker}: {len(data)} clean numerical records")
                    all_data[ticker] = data
                    
                except Exception as e:
                    print(f"Error loading {ticker}: {e}")
            else:
                print(f"File not found: {filepath}")
        
        print(f"Total stocks loaded: {len(all_data)}")
        return all_data
    
    def create_classification_sequences(self, data, ticker):
        """Create classification sequences for next-day Long/Short prediction"""
        print(f"Creating classification sequences for {ticker}...")
        
        data = data.copy()
        
        # Calculate next day's return and create labels
        # Label = 1 (Long) if next day's return > threshold, else 0 (Short)
        data['Next_Return'] = data['Close'].pct_change().shift(-1)
        data['Label'] = (data['Next_Return'] > CLASSIFICATION_THRESHOLD).astype(int)
        data = data.dropna()
        
        # Use multiple features for better prediction
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Scale features
        feature_data = data[features].values
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(feature_data)
        
        # Create sequences
        X, y = [], []
        for i in range(SEQUENCE_LENGTH, len(scaled_features)):
            # Use past SEQUENCE_LENGTH days to predict next day
            X.append(scaled_features[i-SEQUENCE_LENGTH:i])
            y.append(data['Label'].iloc[i])
        
        X = np.array(X)
        y = np.array(y)
        
        # Calculate class distribution
        long_count = sum(y)
        short_count = len(y) - long_count
        long_ratio = long_count / len(y)
        
        print(f"  {ticker} classification: {len(X)} sequences")
        print(f"    Long: {long_count} ({long_ratio:.2%}), Short: {short_count} ({1-long_ratio:.2%})")
        
        return X, y, scaler
    
    def split_data(self, X, y):
        """Split data into train and test sets (chronological split)"""
        split_idx = int(len(X) * (1 - TEST_SIZE))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        return X_train, X_test, y_train, y_test
    
    def process_all_stocks(self):
        """Process all stocks and create classification sequences"""
        print("Starting sequence creation for daily classification...")
        
        # Load data
        all_data = self.load_data()
        if not all_data:
            print("No data found! Run data_collector.py first.")
            return None
        
        classification_data = {}
        
        print("\nProcessing stocks for classification:")
        print("=" * 60)
        
        for ticker, data in all_data.items():
            print(f"Processing {ticker}...")
            
            try:
                # Classification sequences for next-day prediction
                X_cls, y_cls, scaler_cls = self.create_classification_sequences(data, ticker)
                X_train_cls, X_test_cls, y_train_cls, y_test_cls = self.split_data(X_cls, y_cls)
                
                classification_data[ticker] = {
                    'X_train': X_train_cls, 'X_test': X_test_cls,
                    'y_train': y_train_cls, 'y_test': y_test_cls,
                    'scaler': scaler_cls,
                    'features': ['Open', 'High', 'Low', 'Close', 'Volume']
                }
                
                print(f"  {ticker} completed successfully\n")
                
            except Exception as e:
                print(f"  Error processing {ticker}: {e}\n")
                continue
        
        # Save datasets
        if classification_data:
            self.save_datasets(classification_data)
        
        return classification_data
    
    def save_datasets(self, classification_data):
        """Save all classification datasets"""
        print("Saving classification datasets...")
        
        # Create directory
        classification_dir = os.path.join(PROCESSED_DATA_DIR, 'classification')
        os.makedirs(classification_dir, exist_ok=True)
        
        # Save classification data for each stock
        for ticker, data in classification_data.items():
            filepath = os.path.join(classification_dir, f'{ticker}.npz')
            np.savez(filepath,
                     X_train=data['X_train'], X_test=data['X_test'],
                     y_train=data['y_train'], y_test=data['y_test'])
            
            # Save scaler
            scaler_path = os.path.join(classification_dir, f'{ticker}_scaler.pkl')
            joblib.dump(data['scaler'], scaler_path)
            
            # Save feature info
            feature_info = {
                'features': data['features'],
                'sequence_length': SEQUENCE_LENGTH,
                'classification_threshold': CLASSIFICATION_THRESHOLD
            }
            feature_path = os.path.join(classification_dir, f'{ticker}_features.json')
            with open(feature_path, 'w') as f:
                json.dump(feature_info, f, indent=2)
        
        # Save overall metadata
        metadata = {
            'task_type': 'classification',
            'prediction_horizon': 'next_day',
            'total_stocks': len(classification_data),
            'stocks_processed': list(classification_data.keys()),
            'sequence_length': SEQUENCE_LENGTH,
            'data_interval': DATA_INTERVAL,
            'data_period': DATA_PERIOD,
            'classification_threshold': CLASSIFICATION_THRESHOLD,
            'total_sequences': sum(data['X_train'].shape[0] for data in classification_data.values()),
            'feature_count': 5,  # Open, High, Low, Close, Volume
            'test_size': TEST_SIZE
        }
        
        metadata_path = os.path.join(PROCESSED_DATA_DIR, 'classification_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Classification datasets saved to {classification_dir}")
        print(f"Total sequences: {metadata['total_sequences']:,}")

def main():
    processor = DataPreprocessor()
    classification_data = processor.process_all_stocks()
    
    if classification_data:
        total_sequences = sum(data['X_train'].shape[0] for data in classification_data.values())
        total_stocks = len(classification_data)
        
        print("\n" + "=" * 60)
        print("DAILY CLASSIFICATION PROCESSING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Stocks processed: {total_stocks}")
        print(f"Total sequences: {total_sequences:,}")
        print(f"Sequence length: {SEQUENCE_LENGTH} days")
        print(f"Prediction: Next-day Long/Short (threshold: {CLASSIFICATION_THRESHOLD:.3f})")
        print(f"Features: Open, High, Low, Close, Volume")
        print(f"Perfect for distributed training with {total_stocks} stocks!")
    else:
        print("Processing failed!")

if __name__ == "__main__":
    main()