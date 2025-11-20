# COMP4651-stock-prediction-website
Simple Website for COMP4651 Project Demo

How to use:
1. Enter the Stock Ticker(s) you want to predict. 
2. Select which model to use for prediction.
3. View the predicted stock price output.

Details:
1. Simple Model: 
  - MLPClassifier trained on basic features such as 'Open', 'High', 'Low', 'Close', 'Volume'.
  - Trained on several big name stocks such as AAPL and AMZN due to their market dominance and overall representation.

2. Enhanced Model:
  - RandomForest classifier using additional library pandas_ta, a library designed for technical analysis of financial market data,   
    which provides over 150 technical indicators like Stochastic and Bollinger Bands which can allow a more complex model to 
    have a better understanding of the financial markets.
  - Trained on more big name stocks like MSFT and GOOGL, and even ETFs like QQQ and SPY which provide a more holistic view of the
    whole financial market.
  