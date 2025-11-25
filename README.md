# COMP4651-stock-prediction-website
Simple Website for COMP4651 Project Demo

How to use:
1. Enter the Stock Ticker(s) you want to predict. 
2. Select which model to use for prediction.
3. View the predicted stock price output.

Details:
- app_debug.py is just a simple web app we used to debug whether we were getting the right financial info/models loaded.
- app_Vertex.py is a skeleton for deploying a web app that calls the models that have been deployed onto Vertex AI and are exposed via API endpoints. 
- app_yfinance.py was originally the demo we wanted to use, but when deploying onto Render (platform for web service hosting) Yahoo Finance were rate limiting us from scraping their data.
- app.py is the demo app we used during presentation as a proof-of-concept for ML training pipeline on the cloud. We switched from Yahoo Finance to Alpha Vantage to provide us with the financial data. To host and run on Render, please set an ENVIRONMENT VARIABLE: ALPHA_VANTAGE_API_KEY as your own free API access key from https://www.alphavantage.co/. The free API only allows certain "free" APIs and the one we used was TIME_SERIES_DAILY.

* To use the trained models, switch to main branch and use the app.py there
1. Simple Model: 
  - MLPClassifier trained on basic features such as 'Open', 'High', 'Low', 'Close', 'Volume'.
  - Trained on several big name stocks such as AAPL and AMZN due to their market dominance and overall representation.

2. Enhanced Model:
  - RandomForest classifier using additional library pandas_ta, a library designed for technical analysis of financial market data,   
    which provides over 150 technical indicators like Stochastic and Bollinger Bands which can allow a more complex model to 
    have a better understanding of the financial markets.
  - Trained on more big name stocks like MSFT and GOOGL, and even ETFs like QQQ and SPY which provide a more holistic view of the
    whole financial market.
  