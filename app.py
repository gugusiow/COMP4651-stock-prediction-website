import random
import requests
import os
from flask import Flask, render_template, request, jsonify
from alpha_vantage.timeseries import TimeSeries
# from alpha_vantage.symbolsearch import SymbolSearch

import logging

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Set your Alpha Vantage API key in an environment variable ALPHA_VANTAGE_API_KEY
ALPHA_VANTAGE_API_KEY = os.environ.get('ALPHA_VANTAGE_API_KEY')

ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='json')

# ss = SymbolSearch(key=ALPHA_VANTAGE_API_KEY, output_format='json')

def get_company_name(symbol):
    try:
        url = f'https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={symbol}&apikey={ALPHA_VANTAGE_API_KEY}'
        # response = ss.query(symbol)
        response = requests.get(url)
        data=response.json()
        best_matches = data.get('bestMatches', [])
        # Find exact ticker match (case insensitive) or fallback to first match
        for match in best_matches:
            if match.get('1. symbol', '').upper() == symbol.upper():
                return match.get('2. name', symbol)
        if best_matches:
            return best_matches[0].get('2. name', symbol)

        return symbol  # fallback if no matches
    except Exception as e:
        print(f"Error fetching company name for {symbol}: {e}")
        return symbol

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True) or {}
    tickers = data.get('tickers', [])

    logging.info(f"Received tickers for prediction: {tickers}")

    if not isinstance(tickers, list) or not tickers:
        return jsonify({'error': 'Tickers must be a non-empty list'}), 400

    valid_tickers = []
    for raw in tickers[:8]:
        cleaned = str(raw).strip().upper()
        if cleaned and len(cleaned) <= 6:
            valid_tickers.append(cleaned)

    results = []

    for ticker in valid_tickers:
        try:
            logging.info(f"Fetching data from Alpha Vantage for ticker: {ticker}")
            # Fetch daily adjusted close price time series for last 100 days
            # data, meta_data = ts.get_daily_adjusted(symbol=ticker, outputsize='compact')  # daily_adjusted is a premium endpoint lmao
            data, meta_data = ts.get_daily(symbol=ticker, outputsize='compact')

            logging.debug(f"Alpha Vantage response metadata for {ticker}: {meta_data}")
            logging.debug(f"Sample data for {ticker}: {list(data.items())[:3]}")

            last_refreshed = meta_data['3. Last Refreshed']
            last_close = data[last_refreshed]['4. close']
            last_price = float(last_close)

            logging.info(f"{ticker} - last refreshed: {last_refreshed}, closing price: {last_close}")

            # Mock prediction: ±3%
            mock_pred = last_price * (1 + random.uniform(-0.03, 0.03))
            confidence = round(random.uniform(60, 99), 2)
            change = (mock_pred - last_price) / last_price * 100

            company_name = get_company_name(ticker)
            # company_name = ticker  # fallback to ticker itself

            model = data.get('model', 'ML')
            if model == "simple":
                model = "Simple Model"
            else:
                model = "Enhanced Model"

            results.append({
                'change': round(change, 2),
                'ticker': ticker,
                'company_name': company_name,
                'current_price': round(last_price, 2),
                'predicted_price': round(mock_pred, 2),
                'confidence': confidence,
                'method': model
            })

        except Exception as e:
            logging.error(f"Error fetching or processing data for {ticker}: {e}")
            # Fallback: Return fixed or dynamically generated mock data
            print(f"API fetch error for {ticker}: {e} - returning mock data")
            last_price = 271.49  # arbitrary mock price
            if ticker == "AAPL":
                last_price = 271.49
            elif ticker == "META":
                last_price = 594.25
            elif ticker == "MSFT":
                last_price = 472.12
            elif ticker == "JD":
                last_price = 28.93
            elif ticker == "NVDA":
                last_price = 178.88
            elif ticker == "AVGO":
                last_price = 340.20
            
            mock_pred = last_price * (1 + random.uniform(-0.03, 0.02))
            confidence = round(random.uniform(60, 99), 2)
            change = (mock_pred - last_price) / last_price * 100
            company_name = ""

            model = data.get('model', 'ML')
            if model == "simple":
                model = "Simple Model"
            else:
                model = "Enhanced Model"

            results.append({
                'change': round(change, 2),
                'ticker': ticker,
                'company_name': company_name,
                'current_price': round(last_price, 2),
                'predicted_price': round(mock_pred, 2),
                'confidence': confidence,
                'method': model
            })

    return jsonify({'predictions': results}), 200

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
