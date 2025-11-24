from flask import Flask, render_template, request, jsonify
import os
import logging
import google.auth
import requests
from google.auth.transport.requests import Request
from google.oauth2 import service_account
import json

app = Flask(__name__)

# Your Vertex AI endpoint resource ID, e.g.,
# projects/{project}/locations/{location}/endpoints/{endpoint_id}
VERTEX_AI_ENDPOINT = os.environ.get("VERTEX_AI_ENDPOINT")

# Load service account credentials from env var JSON string
def get_access_token():
    json_creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if not json_creds:
        raise Exception("Missing Google credentials in environment variable!")
    creds_info = json.loads(json_creds)
    credentials = service_account.Credentials.from_service_account_info(
        creds_info,
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    credentials.refresh(Request())
    return credentials.token

# Obtain Google Cloud OAuth 2.0 token for authentication
def get_access_token():
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    credentials.refresh(Request())
    return credentials.token

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        tickers = data.get('tickers', [])
        
        if not tickers:
            return jsonify({'error': 'No tickers provided'}), 400
        
        # Clean and cap tickers to 8 max, uppercase and strip spaces
        valid_tickers = []
        for ticker in tickers[:8]:
            cleaned = str(ticker).strip().upper()
            if cleaned and len(cleaned) <= 6:
                valid_tickers.append(cleaned)

        if not valid_tickers:
            return jsonify({'error': 'No valid tickers'}), 400

        # Prepare the instance format expected by your Vertex AI model
        # Adapt this depending on your model input format
        instances = [{"tickers": valid_tickers}]
        
        access_token = get_access_token()
        
        prediction_url = f"https://{VERTEX_AI_ENDPOINT}:predict"
        
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }
        
        response = requests.post(
            prediction_url,
            headers=headers,
            json={"instances": instances}
        )
        
        if response.status_code != 200:
            logging.error(f"Vertex AI API error: {response.text}")
            return jsonify({'error': 'Failed to get prediction from Vertex AI'}), 500
        
        prediction_response = response.json()
        
        return jsonify({
            'predictions': prediction_response.get('predictions', []),
            'model_endpoint': VERTEX_AI_ENDPOINT
        })

    except Exception as e:
        logging.error(f"Prediction endpoint error: {e}")
        return jsonify({'error': 'Service temporarily unavailable'}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
