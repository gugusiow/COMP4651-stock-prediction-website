import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockPredictor:
    def __init__(self):
        self.sequence_length = 60
        self.model = None
        self.scaler = None
        self.features = ['Open', 'High', 'Low', 'Close', 'Volume']
        
    def load_model(self, model_path="models/model.pkl", scaler_path="models/scaler.pkl"):
        """Load trained model and scaler"""
        try:
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                logger.info("Model loaded successfully")
            else:
                logger.warning("No trained model found, using algorithmic predictions")
                
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                logger.info("Scaler loaded successfully")
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None
            self.scaler = None
    
    def fetch_recent_data(self, ticker, days=90):
        """Fetch recent stock data for prediction"""
        try:
            logger.info(f"Fetching data for {ticker}")
            stock = yf.Ticker(ticker)
            data = stock.history(period=f"{days}d", interval="1d")
            
            if data.empty:
                logger.error(f"No data found for {ticker}")
                return None
            
            # Keep only the 5 basic features
            available_cols = [col for col in self.features if col in data.columns]
            if len(available_cols) < 5:
                logger.error(f"Missing basic features for {ticker}")
                return None
                
            data = data[available_cols]
            data = data.dropna()
            
            logger.info(f"Fetched {len(data)} records for {ticker}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return None
    
    def create_sequence(self, data):
        """Create sequence for prediction using only basic features"""
        try:
            # Use only the 5 basic features
            feature_data = data[self.features].values
            
            # We need to scale the entire sequence properly
            # If we have a scaler, it expects 300 features (60×5)
            # So we need to create a full sequence first, then scale
            
            # Take the most recent sequence
            if len(feature_data) >= self.sequence_length:
                sequence = feature_data[-self.sequence_length:]
            else:
                # Pad with zeros if not enough data
                padding = np.zeros((self.sequence_length - len(feature_data), len(self.features)))
                sequence = np.vstack([padding, feature_data])
            
            # Flatten the sequence for the scaler (1, 300)
            sequence_flat = sequence.reshape(1, -1)
            
            # Scale using the trained scaler if available
            if self.scaler and hasattr(self.scaler, 'transform'):
                try:
                    sequence_scaled = self.scaler.transform(sequence_flat)
                except ValueError as e:
                    logger.warning(f"Scaler error: {e}. Using local scaling.")
                    # Fallback: scale locally
                    local_scaler = MinMaxScaler()
                    sequence_scaled = local_scaler.fit_transform(sequence_flat)
            else:
                # Scale locally if no scaler available
                local_scaler = MinMaxScaler()
                sequence_scaled = local_scaler.fit_transform(sequence_flat)
            
            return sequence_scaled
            
        except Exception as e:
            logger.error(f"Error creating sequence: {e}")
            return None
    
    def predict(self, ticker):
        """Make prediction for a single stock using basic features only"""
        try:
            # Fetch recent data
            data = self.fetch_recent_data(ticker)
            if data is None or len(data) < 30:
                return self._create_mock_prediction(ticker)
            
            # Get current price
            current_price = data['Close'].iloc[-1]
            
            if self.model and hasattr(self.model, 'predict'):
                # Create sequence for prediction
                sequence_scaled = self.create_sequence(data)
                if sequence_scaled is None:
                    raise ValueError("Could not create prediction sequence")
                
                # Make prediction (sequence is already flattened and scaled)
                prediction = self.model.predict(sequence_scaled)[0]
                
                # Get confidence from probabilities
                if hasattr(self.model, 'predict_proba'):
                    prediction_proba = self.model.predict_proba(sequence_scaled)[0]
                    confidence = max(prediction_proba)
                else:
                    confidence = 0.7
                
                # Convert prediction to price movement (1=up, 0=down)
                base_change = 0.015 if prediction == 1 else -0.015
                
            else:
                # Use algorithmic prediction based on recent trend
                base_change = self._calculate_recent_trend(data)
                confidence = 0.6 + min(abs(base_change) * 10, 0.3)
            
            # Calculate predicted price
            predicted_price = current_price * (1 + base_change)
            
            return {
                'ticker': ticker,
                'current_price': round(current_price, 2),
                'predicted_price': round(predicted_price, 2),
                'change': round(base_change * 100, 2),
                'confidence': round(min(confidence * 100, 95), 1),
                'method': 'ML Model' if self.model else 'Algorithm'
            }
            
        except Exception as e:
            logger.error(f"Prediction error for {ticker}: {e}")
            return self._create_mock_prediction(ticker)
    
    def _calculate_recent_trend(self, data, lookback_days=10):
        """Calculate recent price trend for fallback prediction"""
        try:
            if len(data) < lookback_days:
                lookback_days = len(data)
            
            recent_closes = data['Close'].tail(lookback_days)
            if len(recent_closes) < 2:
                return 0
            
            # Calculate simple trend
            first_price = recent_closes.iloc[0]
            last_price = recent_closes.iloc[-1]
            trend = (last_price - first_price) / first_price
            
            # Dampen the trend
            dampened_trend = trend * 0.5
            
            return min(max(dampened_trend, -0.04), 0.04)  # Cap at ±4%
            
        except Exception as e:
            logger.error(f"Error calculating trend: {e}")
            return 0
    
    def _create_mock_prediction(self, ticker):
        """Create mock prediction when real prediction fails"""
        import random
        
        # Realistic base prices for common stocks
        stock_base_prices = {
            'AAPL': 268.56, 'MSFT': 487.12, 'TSLA': 403.99, 
            'AMZN': 222.69, 'JPM': 303.27, 'JNJ': 202.51, 'XOM': 117.35
        }
        
        base_price = stock_base_prices.get(ticker, 100)
        change = random.uniform(-2.0, 2.0)
        
        return {
            'ticker': ticker,
            'current_price': round(base_price, 2),
            'predicted_price': round(base_price * (1 + change/100), 2),
            'change': round(change, 2),
            'confidence': round(random.uniform(60, 75), 1),
            'method': 'Mock Data'
        }

# Global predictor instance
predictor = StockPredictor()