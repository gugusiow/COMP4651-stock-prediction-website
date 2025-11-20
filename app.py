from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        tickers = data.get('tickers', [])
        
        # TODO: Replace with your actual ML model prediction logic
        # For now, returning mock data
        mock_predictions = []
        for ticker in tickers[:5]:  # Limit to 5 tickers for demo
            mock_predictions.append({
                'ticker': ticker,
                'current_price': round(150.25 + len(ticker), 2),
                'predicted_price': round(152.80 + len(ticker), 2),
                'change': round(1.7 + (len(ticker) * 0.1), 2),
                'confidence': round(78.5 + (len(ticker) * 0.5), 1)
            })
        
        return jsonify({'predictions': mock_predictions})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Health check route for Render
@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)