import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import joblib
import os

def fetch_training_data():
    """Fetch sample data for training a simple model"""
    tickers = ['AAPL', 'MSFT', 'TSLA', 'JPM', 'AMZN', 'XOM', 'JNJ']
    all_data = []
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period="2y", interval="1d")
            
            if not data.empty:
                # Calculate features
                data['Returns'] = data['Close'].pct_change()
                data['MA_5'] = data['Close'].rolling(5).mean()
                data['MA_20'] = data['Close'].rolling(20).mean()
                data['Volume_MA'] = data['Volume'].rolling(10).mean()
                data['Target'] = (data['Returns'].shift(-1) > 0).astype(int)
                
                data = data.dropna()
                all_data.append(data)
                print(f"Fetched {len(data)} records for {ticker}")
                
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
    
    return pd.concat(all_data) if all_data else None

def train_simple_model():
    """Train a simple Random Forest model"""
    print("Fetching training data...")
    data = fetch_training_data()
    
    if data is None:
        print("No data fetched, creating mock model...")
        # Create a mock model for demonstration
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=1000, n_features=5, random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Create mock scaler
        scaler = MinMaxScaler()
        scaler.fit(np.random.rand(10, 5))
        
    else:
        # Prepare features
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_5', 'MA_20', 'Volume_MA']
        X = data[features]
        y = data['Target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        print(f"Model trained - Train score: {train_score:.3f}, Test score: {test_score:.3f}")
    
    # Save model and scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Model and scaler saved to models/ directory")

if __name__ == "__main__":
    train_simple_model()