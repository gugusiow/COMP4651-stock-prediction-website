from flask import Flask, render_template, request, jsonify
import os
import random

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        tickers = data.get('tickers', [])
        
        # Simple mock predictions without ML dependencies
        mock_predictions = []
        for ticker in tickers[:5]:  # Limit to 5 tickers
            base_price = 100 + len(ticker) * 5
            change_percent = random.uniform(-5, 5)
            predicted_price = base_price * (1 + change_percent/100)
            
            mock_predictions.append({
                'ticker': ticker,
                'current_price': round(base_price, 2),
                'predicted_price': round(predicted_price, 2),
                'change': round(change_percent, 2),
                'confidence': round(random.uniform(60, 95), 1)
            })
        
        return jsonify({'predictions': mock_predictions})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)