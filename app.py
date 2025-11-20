from flask import Flask, render_template, request, jsonify
import os
import logging
from ml_pipeline.model_predictor import predictor as simple_predictor
from ml_pipeline.enhanced_predictor import enhanced_predictor

# Initialize both predictors
simple_predictor.load_model()
enhanced_predictor.load_model()

app = Flask(__name__)

# Store the current model selection
current_model = "enhanced"  # default to enhanced

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        tickers = data.get('tickers', [])
        model_type = data.get('model_type', 'enhanced')  # Get model type from request
        
        global current_model
        current_model = model_type  # Update current model selection
        
        if not tickers:
            return jsonify({'error': 'No tickers provided'}), 400
        
        # Clean tickers
        valid_tickers = []
        for ticker in tickers[:8]:  # Limit to 8 tickers
            cleaned = str(ticker).strip().upper()
            if cleaned and len(cleaned) <= 6:
                valid_tickers.append(cleaned)
        
        if not valid_tickers:
            return jsonify({'error': 'No valid tickers'}), 400
        
        # Select the appropriate predictor
        if model_type == 'simple':
            predictor = simple_predictor
            model_name = "Simple ML"
        else:
            predictor = enhanced_predictor
            model_name = "Enhanced ML"
        
        # Make predictions
        predictions = []
        for ticker in valid_tickers:
            try:
                prediction = predictor.predict(ticker)
                prediction['model_used'] = model_name
                predictions.append(prediction)
            except Exception as e:
                logging.error(f"Error predicting {ticker} with {model_name}: {e}")
                predictions.append({
                    'ticker': ticker,
                    'error': 'Prediction failed',
                    'current_price': 0,
                    'predicted_price': 0,
                    'change': 0,
                    'confidence': 0,
                    'method': 'Error',
                    'model_used': model_name
                })
        
        return jsonify({
            'predictions': predictions,
            'model_used': model_name
        })
    
    except Exception as e:
        logging.error(f"Prediction endpoint error: {e}")
        return jsonify({'error': 'Service temporarily unavailable'}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy', 
        'simple_model_loaded': simple_predictor.model is not None,
        'enhanced_model_loaded': enhanced_predictor.model is not None,
        'current_model': current_model
    })

@app.route('/switch-model', methods=['POST'])
def switch_model():
    """Endpoint to switch models"""
    try:
        data = request.get_json()
        model_type = data.get('model_type', 'enhanced')
        
        global current_model
        current_model = model_type
        
        return jsonify({
            'status': 'success', 
            'current_model': current_model,
            'message': f'Switched to {model_type} model'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/model-info')
def model_info():
    """Get information about available models"""
    simple_info = {
        'loaded': simple_predictor.model is not None,
        'type': 'Simple ML',
        'features': 'OHLCV + Basic Sequences',
        'sequence_length': getattr(simple_predictor, 'sequence_length', 60)
    }
    
    enhanced_info = {
        'loaded': enhanced_predictor.model is not None,
        'type': 'Enhanced ML', 
        'features': 'Technical Indicators + Advanced Features',
        'sequence_length': getattr(enhanced_predictor, 'sequence_length', 20)
    }
    
    return jsonify({
        'simple_model': simple_info,
        'enhanced_model': enhanced_info,
        'current_model': current_model
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)