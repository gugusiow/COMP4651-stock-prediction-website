# train_sequence_model.py
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import yfinance as yf
import joblib
import os

def fetch_stock_data(tickers, period="2y", sequence_length=60):
    """Fetch stock data and create sequences"""
    all_sequences = []
    all_labels = []
    
    for ticker in tickers:
        try:
            print(f"Fetching {ticker}...")
            stock = yf.Ticker(ticker)
            data = stock.history(period=period, interval="1d")
            
            if data.empty or len(data) < sequence_length + 10:
                print(f"  Insufficient data for {ticker}")
                continue
            
            # Use only basic features
            features = ['Open', 'High', 'Low', 'Close', 'Volume']
            data = data[features].dropna()
            
            # Create sequences and labels
            sequences, labels = create_sequences(data, sequence_length)
            
            if len(sequences) > 0:
                all_sequences.extend(sequences)
                all_labels.extend(labels)
                print(f"  {ticker}: {len(sequences)} sequences")
            
        except Exception as e:
            print(f"  Error processing {ticker}: {e}")
    
    return np.array(all_sequences), np.array(all_labels)

def create_sequences(data, sequence_length=60):
    """Create sequences and labels for classification"""
    sequences = []
    labels = []
    
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    values = data[features].values
    
    for i in range(sequence_length, len(values) - 1):
        # Sequence of past prices
        sequence = values[i-sequence_length:i]
        
        # Label: 1 if next day's return > 0.001 (0.1%), else 0
        current_close = values[i, 3]  # Close price
        next_close = values[i+1, 3]   # Next day's close
        return_tomorrow = (next_close - current_close) / current_close
        
        label = 1 if return_tomorrow > 0.001 else 0
        
        sequences.append(sequence)
        labels.append(label)
    
    return sequences, labels

def train_sequence_model():
    """Train model using sequence data"""
    print("Training Sequence-Based OHLCV Model")
    print("=" * 50)
    
    # Use multiple stocks for training
    tickers = ['AAPL', 'MSFT', 'TSLA', 'GOOGL', 'AMZN', 'META', 'NVDA', 'JPM', 'JNJ', 'XOM']
    sequence_length = 60
    
    print("Fetching training data...")
    X, y = fetch_stock_data(tickers, sequence_length=sequence_length)
    
    if len(X) == 0:
        print("No real data fetched. Creating realistic synthetic data...")
        X, y = create_synthetic_data()
    
    print(f"Training data: {X.shape} sequences, {len(y)} labels")
    print(f"Sequence shape: {X[0].shape}")
    
    # Flatten sequences for MLP (samples, 60 * 5 = 300 features)
    X_flat = X.reshape(X.shape[0], -1)
    print(f"Flattened data: {X_flat.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_flat, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features - this scaler will expect 300 features!
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Scaling completed - scaler expects 300 features")
    
    # Train MLP model
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=32,
        learning_rate='adaptive',
        max_iter=50,  # Reduced for faster training
        random_state=42,
        early_stopping=True,
        n_iter_no_change=5,
        verbose=True
    )
    
    print("Training model...")
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    print(f"\nModel Performance:")
    print(f"Train Accuracy: {train_acc:.3f}")
    print(f"Test Accuracy:  {test_acc:.3f}")
    
    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    class_dist = dict(zip(['Down', 'Up'], counts))
    print(f"Class distribution: {class_dist}")
    
    # Save model and scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # Save configuration
    config = {
        'features': ['Open', 'High', 'Low', 'Close', 'Volume'],
        'sequence_length': sequence_length,
        'classification_threshold': 0.001,
        'training_samples': len(X),
        'input_dimension': X_flat.shape[1],
        'test_accuracy': float(test_acc)
    }
    
    import json
    with open('models/config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nModel saved to models/ directory")
    print(f"Features: {config['features']}")
    print(f"Sequence length: {config['sequence_length']}")
    print(f"Input dimension: {config['input_dimension']} (60×5=300)")
    print(f"Test accuracy: {config['test_accuracy']:.3f}")

def create_synthetic_data(n_samples=2000, sequence_length=60, n_features=5):
    """Create realistic synthetic stock data for testing"""
    print("Creating synthetic stock data...")
    
    np.random.seed(42)
    
    # Create realistic price patterns
    X = []
    y = []
    
    for i in range(n_samples):
        # Start with a base price around 100
        base_price = 100 + np.random.uniform(-20, 20)
        
        # Create a random walk for prices
        prices = [base_price]
        for j in range(sequence_length * 2):  # Generate extra for volatility
            change = np.random.normal(0, 0.02)  # 2% daily volatility
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1))  # Ensure positive price
        
        # Use the last sequence_length prices
        price_sequence = prices[-sequence_length:]
        
        # Create OHLCV data from price sequence
        sequence = []
        for price in price_sequence:
            # Generate realistic OHLC from close price
            volatility = np.random.uniform(0.005, 0.02)
            open_price = price * (1 + np.random.normal(0, volatility))
            high = max(open_price, price) * (1 + abs(np.random.normal(0, volatility/2)))
            low = min(open_price, price) * (1 - abs(np.random.normal(0, volatility/2)))
            close = price
            volume = np.random.uniform(1e6, 1e7)
            
            sequence.append([open_price, high, low, close, volume])
        
        X.append(sequence)
        
        # Random label for synthetic data
        label = np.random.choice([0, 1], p=[0.5, 0.5])
        y.append(label)
    
    return np.array(X), np.array(y)

if __name__ == "__main__":
    train_sequence_model()