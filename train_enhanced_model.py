import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import yfinance as yf
import joblib
import os
import pandas_ta as ta

def add_technical_indicators(data):
    """Add comprehensive technical indicators using pandas_ta with proper column handling"""
    df = data.copy()
    
    # Basic price features
    df['Returns'] = df['Close'].pct_change()
    df['Price_Range'] = (df['High'] - df['Low']) / df['Close']
    df['Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    
    # Moving averages (simple ones first)
    df['MA_5'] = df['Close'].rolling(5).mean()
    df['MA_10'] = df['Close'].rolling(10).mean()
    df['MA_20'] = df['Close'].rolling(20).mean()
    df['MA_50'] = df['Close'].rolling(50).mean()
    df['MA_5_20_Ratio'] = df['MA_5'] / df['MA_20']

    try:
        # RSI
        rsi = ta.rsi(df['Close'], length=14)
        if rsi is not None:
            df['RSI_14'] = rsi
    except Exception as e:
        print(f"  RSI error: {e}")
        df['RSI_14'] = 50  # Default value

    try:
        # MACD
        macd_result = ta.macd(df['Close'], fast=12, slow=26, signal=9)
        if macd_result is not None:
            # Get the first column (MACD line)
            macd_columns = [col for col in macd_result.columns if 'MACD_' in col]
            if macd_columns:
                df['MACD'] = macd_result[macd_columns[0]]
    except Exception as e:
        print(f"  MACD error: {e}")
        df['MACD'] = 0

    try:
        # Bollinger Bands - use different approach
        bb_result = ta.bbands(df['Close'], length=20, std=2)
        if bb_result is not None:
            # Find the actual column names
            bb_columns = bb_result.columns
            upper_col = [col for col in bb_columns if 'BBU' in col or 'upper' in col.lower()]
            lower_col = [col for col in bb_columns if 'BBL' in col or 'lower' in col.lower()]
            middle_col = [col for col in bb_columns if 'BBM' in col or 'middle' in col.lower()]
            
            if upper_col and lower_col:
                df['BB_Upper'] = bb_result[upper_col[0]]
                df['BB_Lower'] = bb_result[lower_col[0]]
                if middle_col:
                    df['BB_Middle'] = bb_result[middle_col[0]]
                    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    except Exception as e:
        print(f"  Bollinger Bands error: {e}")

    try:
        # Stochastic
        stoch_result = ta.stoch(df['High'], df['Low'], df['Close'], k=14, d=3)
        if stoch_result is not None:
            stoch_columns = stoch_result.columns
            k_col = [col for col in stoch_columns if 'STOCHk' in col or 'k_' in col]
            if k_col:
                df['Stoch_K'] = stoch_result[k_col[0]]
    except Exception as e:
        print(f"  Stochastic error: {e}")

    # Volume indicators
    df['Volume_MA'] = df['Volume'].rolling(10).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
    
    # Volatility
    df['Volatility_20'] = df['Returns'].rolling(20).std()
    
    # Price position features
    df['Price_vs_High_20'] = df['Close'] / df['High'].rolling(20).max()
    df['Price_vs_Low_20'] = df['Close'] / df['Low'].rolling(20).min()
    
    # Ensure we have some basic features even if technical indicators fail
    required_basic_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 
                              'MA_5', 'MA_20', 'MA_5_20_Ratio', 'Volume_Ratio']
    
    # Add default values for missing technical features
    technical_features = ['RSI_14', 'MACD', 'BB_Position', 'Stoch_K']
    for feature in technical_features:
        if feature not in df.columns:
            if feature == 'RSI_14':
                df[feature] = 50
            elif feature == 'MACD':
                df[feature] = 0
            elif feature == 'BB_Position':
                df[feature] = 0.5
            elif feature == 'Stoch_K':
                df[feature] = 50
    
    # Drop the missing values
    df = df.dropna()
    
    return df

def create_enhanced_features(data, sequence_length=15):
    """Create feature sequences with technical indicators"""
    # Define feature columns - use a consistent set
    feature_columns = [
        # Basic features
        'Open', 'High', 'Low', 'Close', 'Volume', 'Returns',
        # Moving averages
        'MA_5', 'MA_20', 'MA_5_20_Ratio',
        # Technical indicators
        'RSI_14', 'MACD', 'BB_Position', 'Stoch_K',
        # Volume and volatility
        'Volume_Ratio', 'Volatility_20',
        # Price position
        'Price_vs_High_20', 'Price_vs_Low_20'
    ]
    
    # Use only features that actually exist in the data
    available_features = [f for f in feature_columns if f in data.columns]
    
    print(f"  Using {len(available_features)} features: {available_features}")
    
    sequences = []
    labels = []
    
    values = data[available_features].values
    
    for i in range(sequence_length, len(values) - 5):  # Predict 5 days ahead
        sequence = values[i-sequence_length:i]
        
        # We set the target to be > 2% gain
        current_price = data['Close'].iloc[i]
        future_idx = min(i + 5, len(data) - 1)
        future_price = data['Close'].iloc[future_idx]
        future_return = (future_price - current_price) / current_price
        
        # Binary classification, label is 1 if > 2% gain in 5 days, else 0
        label = 1 if future_return > 0.02 else 0
        
        sequences.append(sequence)
        labels.append(label)
    
    return np.array(sequences), np.array(labels)

def fetch_enhanced_data(tickers, period="2y"):
    """Fetch data with enhanced features - simplified approach"""
    all_sequences = []
    all_labels = []
    
    successful_tickers = 0
    
    for ticker in tickers:
        try:
            print(f"Processing {ticker}...")
            stock = yf.Ticker(ticker)
            data = stock.history(period=period, interval="1d")
            
            if data.empty or len(data) < 80:
                print(f"  Insufficient data for {ticker}")
                continue
            
            # Add technical indicators
            data = add_technical_indicators(data)
            
            if len(data) < 40:
                print(f"  Not enough data after processing for {ticker}")
                continue
            
            # Create sequences
            sequences, labels = create_enhanced_features(data, sequence_length=15)
            
            if len(sequences) > 20:
                all_sequences.extend(sequences)
                all_labels.extend(labels)
                successful_tickers += 1
                print(f"   {ticker}: {len(sequences)} sequences")
            else:
                print(f"   {ticker}: Not enough sequences")
            
        except Exception as e:
            print(f"   Error with {ticker}: {str(e)[:100]}...")
    
    print(f"\nSuccessfully processed {successful_tickers}/{len(tickers)} tickers")
    
    if len(all_sequences) == 0:
        print("No real data processed. Creating synthetic data for demonstration...")
        return create_synthetic_data()
    
    return np.array(all_sequences), np.array(all_labels)

def create_synthetic_data(n_samples=1200, sequence_length=15, n_features=16):
    print("Creating mock stock data for demonstration...")
    
    np.random.seed(42)
    
    X = []
    y = []
    
    for i in range(n_samples):
        # Create realistic price pattern with trends
        base_price = 100 + np.random.uniform(-30, 30)
        prices = [base_price]
        
        # Add some random trend/momentum
        trend_direction = np.random.choice([-1, 1])
        trend_strength = np.random.uniform(0.001, 0.003)
        
        for j in range(sequence_length * 2):
            # Random walk with slight trend
            random_component = np.random.normal(0, 0.015)
            trend_component = trend_direction * trend_strength
            change = random_component + trend_component
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1))
        
        price_sequence = prices[-sequence_length:]
        
        # Create feature sequence
        sequence = []
        for k, price in enumerate(price_sequence):
            open_price = price * (1 + np.random.normal(0, 0.008))
            high = max(open_price, price) * (1 + abs(np.random.normal(0, 0.006)))
            low = min(open_price, price) * (1 - abs(np.random.normal(0, 0.006)))
            close = price
            volume = np.random.lognormal(15, 0.8)  # Realistic volume
            returns = np.random.normal(0, 0.012)
            
            # Technical indicators with realistic relationships
            ma_5 = np.mean(price_sequence[max(0, k-4):k+1]) if k >= 4 else price
            ma_20 = np.mean(price_sequence[max(0, k-19):k+1]) if k >= 19 else price
            ma_ratio = ma_5 / ma_20 if ma_20 != 0 else 1
            
            # RSI based on price momentum
            if k >= 13:
                gains = max(0, price_sequence[k] - price_sequence[k-1])
                losses = max(0, price_sequence[k-1] - price_sequence[k])
                avg_gain = np.mean([max(0, price_sequence[j] - price_sequence[j-1]) for j in range(max(1, k-13), k+1)])
                avg_loss = np.mean([max(0, price_sequence[j-1] - price_sequence[j]) for j in range(max(1, k-13), k+1)])
                rsi = 100 - (100 / (1 + (avg_gain / avg_loss))) if avg_loss != 0 else 50
            else:
                rsi = 50
                
            # MACD-like oscillator
            macd = np.random.normal(0, 0.8)
            
            # Bollinger Band position
            bb_position = np.random.uniform(0.2, 0.8)
            
            # Stochastic
            stoch_k = np.random.uniform(20, 80)
            
            # Volume ratio
            vol_ratio = np.random.lognormal(0, 0.4)
            
            # Volatility
            volatility = np.random.uniform(0.008, 0.025)
            
            # Price position
            price_high_20 = np.random.uniform(0.85, 0.98)
            price_low_20 = np.random.uniform(1.02, 1.15)
            
            features = [open_price, high, low, close, volume, returns, 
                       ma_5, ma_20, ma_ratio, rsi, macd, bb_position, 
                       stoch_k, vol_ratio, volatility, price_high_20, price_low_20]
            
            # Take first n_features
            sequence.append(features[:n_features])
        
        X.append(sequence)
        
        # Determine label based on the trend
        price_change = (price_sequence[-1] - price_sequence[0]) / price_sequence[0]
        if price_change > 0.03:  # Strong upward trend
            label = 1
        elif price_change < -0.03:  # Strong downward trend
            label = 0
        else:
            # Random for sideways
            label = np.random.choice([0, 1], p=[0.5, 0.5])
        
        y.append(label)
    
    return np.array(X), np.array(y)

def train_enhanced_model():
    print("Training Enhanced Stock Prediction Model with Robust Technical Indicators")
    
    # Smaller set of reliable tickers
    tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA',
        'JPM', 'JNJ', 'XOM', 'SPY', 'QQQ'
    ]
    
    print("Fetching and processing data...")
    X, y = fetch_enhanced_data(tickers)
    
    print(f"\n Final Dataset: {X.shape} sequences, {len(y)} labels")
    print(f" Positive class: {y.sum()} ({y.sum()/len(y):.2%})")
    
    # Flatten sequences
    X_flat = X.reshape(X.shape[0], -1)
    print(f" Flattened features: {X_flat.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_flat, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Try multiple models
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
    }
    
    best_model = None
    best_score = 0
    best_name = ""
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=3, scoring='accuracy')
        
        # Test score
        y_pred = model.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        print(f" CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        print(f" Test Accuracy: {test_accuracy:.3f}")
        
        if test_accuracy > best_score:
            best_score = test_accuracy
            best_model = model
            best_name = name
    
    # Feature importance
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = best_model.feature_importances_
        top_indices = np.argsort(feature_importance)[-8:][::-1]
        
        print(f"\n Top 8 Most Important Features:")
        for i, idx in enumerate(top_indices):
            print(f"  {i+1}. Feature {idx}: {feature_importance[idx]:.4f}")
    
    # Detailed evaluation
    y_pred = best_model.predict(X_test_scaled)
    print(f"\n Best Model: {best_name}")
    print(f" Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f" Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    
    # Save model and scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/enhanced_model.pkl')
    joblib.dump(scaler, 'models/enhanced_scaler.pkl')
    
    # Save configuration
    config = {
        'features_used': X_flat.shape[1],
        'sequence_length': 15,
        'prediction_horizon': 5,
        'target_threshold': 0.02,
        'training_samples': len(X),
        'test_accuracy': float(best_score),
        'model_type': best_name
    }
    
    import json
    with open('models/enhanced_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nEnhanced model saved successfully!")
    print(f"Test Accuracy: {best_score:.3f}")
    print(f"Model Type: {best_name}")
    print(f"Training Samples: {len(X):,}")
    print(f"Sequence Length: 15 days")
    print(f"Features: {X_flat.shape[1]}")

if __name__ == "__main__":
    train_enhanced_model()