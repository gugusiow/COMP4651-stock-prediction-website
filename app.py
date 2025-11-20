from flask import Flask, render_template, request, jsonify
import os
import logging
from ml_pipeline.model_predictor import predictor

# Initialize the predictor
predictor.load_model()

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        tickers = data.get('tickers', [])
        
        # Validate input
        if not tickers:
            return jsonify({'error': 'No tickers provided'}), 400
        
        # Clean and validate tickers
        valid_tickers = []
        for ticker in tickers[:10]:  # Limit to 10 tickers
            cleaned_ticker = str(ticker).strip().upper()
            if cleaned_ticker and len(cleaned_ticker) <= 10:
                valid_tickers.append(cleaned_ticker)
        
        if not valid_tickers:
            return jsonify({'error': 'No valid tickers provided'}), 400
        
        # Make predictions
        predictions = []
        for ticker in valid_tickers:
            try:
                prediction = predictor.predict_single_stock(ticker)
                predictions.append(prediction)
            except Exception as e:
                logging.error(f"Error predicting {ticker}: {e}")
                # Include error in response but continue with other tickers
                predictions.append({
                    'ticker': ticker,
                    'error': f'Prediction failed: {str(e)}',
                    'current_price': 0,
                    'predicted_price': 0,
                    'change': 0,
                    'confidence': 0
                })
        
        return jsonify({'predictions': predictions})
    
    except Exception as e:
        logging.error(f"Prediction endpoint error: {e}")
        return jsonify({'error': 'Prediction service temporarily unavailable'}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': predictor.model is not None})

@app.route('/refresh-model')
def refresh_model():
    """Endpoint to reload the model (useful for updates)"""
    try:
        predictor.load_model()
        return jsonify({'status': 'success', 'model_loaded': predictor.model is not None})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)