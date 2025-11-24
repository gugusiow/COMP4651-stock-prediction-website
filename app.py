from flask import Flask, render_template, request, jsonify
import yfinance as yf
import random
import os
import json
import logging
import time

app = Flask(__name__)

# Cache dictionary to store data and timestamps:
cache = {}
CACHE_TTL_SECONDS = 10 * 60  # 10 minutes cache expiry

def get_cached_stock_data(ticker):
    """Return cached data if fresh, else None."""
    cached = cache.get(ticker)
    if cached:
        data, timestamp = cached
        if time.time() - timestamp < CACHE_TTL_SECONDS:
            return data
        else:
            # Expired cache
            cache.pop(ticker, None)
    return None

def set_cache_stock_data(ticker, data):
    """Store ticker data with current timestamp."""
    cache[ticker] = (data, time.time())

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True) or {}
    tickers = data.get('tickers', [])

    logging.info(f"Received tickers: {tickers}")

    if not isinstance(tickers, list):
        logging.error("tickers received is not a list")
        return jsonify({'error': 'tickers must be a list'}), 400

    if not tickers:
        logging.error("No tickers provided")
        return jsonify({'error': 'No tickers provided'}), 400

    # Clean and cap tickers to 8 max, uppercase and strip spaces
    valid_tickers = []
    for raw in tickers[:8]:
        cleaned = str(raw).strip().upper()
        if cleaned and len(cleaned) <= 6:
            valid_tickers.append(cleaned)

    if not valid_tickers:
        logging.error("No valid tickers after cleaning")
        return jsonify({'error': 'No valid tickers'}), 400

    results = []

    for ticker in valid_tickers:
        try:
            logging.info(f"Fetching data for ticker: {ticker}")
            # Check cache first
            cached_data = get_cached_stock_data(ticker)
            if cached_data:
                hist, info = cached_data
            else:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="1d")
                # close_series = hist.get('Close')
                info = stock.info
                set_cache_stock_data(ticker, (hist, info))

            name = info.get('longName', '')
            close_series = hist.get('Close')

            if close_series is None or close_series.empty:
                logging.warning(f"No price data for ticker {ticker}")
                results.append({
                    'ticker': ticker,
                    'company_name': name,
                    'error': f'No price data for ticker {ticker}'
                })
                continue

            last_price = float(close_series.iloc[-1])

            # Mock prediction: random +/- up to 5% of last price
            mock_pred = last_price * (1 + random.uniform(-0.03, 0.02))
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

            logging.info(f"Processed ticker {ticker}: current_price={last_price}, predicted_price={mock_pred}")

        except Exception as e:
            logging.error(f"Error processing ticker {ticker}: {str(e)}")
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
