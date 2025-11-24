from flask import Flask, render_template, request, jsonify
import yfinance as yf
import random
import os
import json

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True) or {}
    tickers = data.get('tickers', [])

    if not isinstance(tickers, list):
        return jsonify({'error': 'tickers must be a list'}), 400

    if not tickers:
        return jsonify({'error': 'No tickers provided'}), 400

    # Clean and cap tickers to 8 max, uppercase and strip spaces
    valid_tickers = []
    for raw in tickers[:8]:
        cleaned = str(raw).strip().upper()
        if cleaned and len(cleaned) <= 6:
            valid_tickers.append(cleaned)

    if not valid_tickers:
        return jsonify({'error': 'No valid tickers'}), 400

    results = []

    for ticker in valid_tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1d")
            close_series = hist.get('Close')
            info = stock.info
            name = info.get('longName', '')

            if close_series is None or close_series.empty:
                results.append({
                    'ticker': ticker,
                    'company_name': name,
                    'error': f'No price data for ticker {ticker}'
                })
                continue

            last_price = float(close_series.iloc[-1])

            # Mock prediction: random +/- up to 5% of last price
            mock_pred = last_price * (1 + random.uniform(-0.03, 0.03))
            confidence = round(random.uniform(60, 99), 2)  # e.g., 70% to 99%
            change = (mock_pred - last_price)/last_price * 100
            model = data.get('model', 'ML')
            if (model == "simple"):
                model = "Simple Model"
            else: 
                model = "Enhanced Model" 
            results.append({
                'change': round(change, 2),
                'ticker': ticker,
                'company_name': name,
                'current_price': round(last_price, 2),
                'predicted_price': round(mock_pred, 2),
                'confidence': confidence,
                'method': model
            })

        except Exception as e:
            results.append({
                'ticker': ticker,
                'company_name': '',
                'error': str(e)
            })

    # Return all results as JSON
    return jsonify({'predictions': results}), 200

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
