from flask import Flask, render_template, request, jsonify
import os
import logging
import traceback

# Set up logging
logging.basicConfig(level=logging.DEBUG)

try:
    from ml_pipeline.model_predictor import predictor as simple_predictor
    from ml_pipeline.enhanced_predictor import enhanced_predictor
    ML_MODULES_LOADED = True
except ImportError as e:
    logging.error(f"Failed to import ML modules: {e}")
    ML_MODULES_LOADED = False
    # Create dummy predictors for testing
    class DummyPredictor:
        def __init__(self, name):
            self.name = name
            self.model = None
        
        def load_model(self):
            logging.info(f"Loading {self.name} model...")
            self.model = "dummy_model"
        
        def predict(self, ticker):
            return {
                'ticker': ticker,
                'current_price': 100.0,
                'predicted_price': 105.0,
                'change': 5.0,
                'confidence': 85.0,
                'method': 'Dummy',
                'model_used': self.name
            }
    
    simple_predictor = DummyPredictor("Simple ML")
    enhanced_predictor = DummyPredictor("Enhanced ML")

app = Flask(__name__)

# Initialize both predictors
try:
    simple_predictor.load_model()
    enhanced_predictor.load_model()
    logging.info("Models loaded successfully")
except Exception as e:
    logging.error(f"Error loading models: {e}")
    traceback.print_exc()

# Store the current model selection
current_model = "enhanced"  # default to enhanced

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        logging.debug(f"Received prediction request: {data}")
        
        tickers = data.get('tickers', [])
        model_type = data.get('model_type', 'enhanced')
        
        global current_model
        current_model = model_type
        
        if not tickers:
            return jsonify({'error': 'No tickers provided'}), 400
        
        # Clean tickers
        valid_tickers = []
        for ticker in tickers[:8]:
            cleaned = str(ticker).strip().upper()
            if cleaned and len(cleaned) <= 6:
                valid_tickers.append(cleaned)
        
        if not valid_tickers:
            return jsonify({'error': 'No valid tickers'}), 400
        
        logging.info(f"Making predictions for {valid_tickers} using {model_type} model")
        
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
                logging.info(f"Predicting for {ticker} using {model_name}...")
                prediction = predictor.predict(ticker)
                logging.info(f"Raw prediction for {ticker}: {prediction}")
                
                # Convert numpy types to native Python types for JSON serialization
                converted_prediction = {}
                for key, value in prediction.items():
                    if hasattr(value, 'item'):  # Check if it's a numpy type
                        converted_prediction[key] = value.item()  # Convert to native Python type
                    else:
                        converted_prediction[key] = value
                
                # Ensure all required fields are present
                required_fields = ['current_price', 'predicted_price', 'change', 'confidence', 'method']
                for field in required_fields:
                    if field not in converted_prediction:
                        logging.warning(f"Missing field {field} in prediction for {ticker}")
                        converted_prediction[field] = 0 if field != 'method' else 'Unknown'
                
                converted_prediction['model_used'] = model_name
                predictions.append(converted_prediction)
                logging.info(f"Processed prediction for {ticker}: {converted_prediction}")
                
            except Exception as e:
                logging.error(f"Error predicting {ticker} with {model_name}: {e}")
                traceback.print_exc()
                predictions.append({
                    'ticker': ticker,
                    'error': f'Prediction failed: {str(e)}',
                    'current_price': 0,
                    'predicted_price': 0,
                    'change': 0,
                    'confidence': 0,
                    'method': 'Error',
                    'model_used': model_name
                })
        
        response = {
            'predictions': predictions,
            'model_used': model_name
        }
        logging.info(f"Sending response with {len(predictions)} predictions")
        return jsonify(response)
    
    except Exception as e:
        logging.error(f"Prediction endpoint error: {e}")
        traceback.print_exc()
        return jsonify({'error': f'Service error: {str(e)}'}), 500
# def predict():
#     try:
#         data = request.get_json()
#         logging.debug(f"Received prediction request: {data}")
        
#         tickers = data.get('tickers', [])
#         model_type = data.get('model_type', 'enhanced')
        
#         global current_model
#         current_model = model_type
        
#         if not tickers:
#             return jsonify({'error': 'No tickers provided'}), 400
        
#         # Clean tickers
#         valid_tickers = []
#         for ticker in tickers[:8]:
#             cleaned = str(ticker).strip().upper()
#             if cleaned and len(cleaned) <= 6:
#                 valid_tickers.append(cleaned)
        
#         if not valid_tickers:
#             return jsonify({'error': 'No valid tickers'}), 400
        
#         logging.info(f"Making predictions for {valid_tickers} using {model_type} model")
        
#         # Select the appropriate predictor
#         if model_type == 'simple':
#             predictor = simple_predictor
#             model_name = "Simple ML"
#         else:
#             predictor = enhanced_predictor
#             model_name = "Enhanced ML"
        
#         # Make predictions
#         predictions = []
#         for ticker in valid_tickers:
#             try:
#                 logging.debug(f"Predicting for {ticker}")
#                 prediction = predictor.predict(ticker)
#                 prediction['model_used'] = model_name
#                 predictions.append(prediction)
#                 logging.debug(f"Prediction for {ticker}: {prediction}")
#             except Exception as e:
#                 logging.error(f"Error predicting {ticker} with {model_name}: {e}")
#                 traceback.print_exc()
#                 predictions.append({
#                     'ticker': ticker,
#                     'error': 'Prediction failed',
#                     'current_price': 0,
#                     'predicted_price': 0,
#                     'change': 0,
#                     'confidence': 0,
#                     'method': 'Error',
#                     'model_used': model_name
#                 })
        
#         response = {
#             'predictions': predictions,
#             'model_used': model_name
#         }
#         logging.debug(f"Sending response: {response}")
#         return jsonify(response)
    
#     except Exception as e:
#         logging.error(f"Prediction endpoint error: {e}")
#         traceback.print_exc()
#         return jsonify({'error': 'Service temporarily unavailable'}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy', 
        'ml_modules_loaded': ML_MODULES_LOADED,
        'simple_model_loaded': simple_predictor.model is not None,
        'enhanced_model_loaded': enhanced_predictor.model is not None,
        'current_model': current_model
    })

@app.route('/switch-model', methods=['POST'])
def switch_model():
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
    app.run(host='0.0.0.0', port=port, debug=True)