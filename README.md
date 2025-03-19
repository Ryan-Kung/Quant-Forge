# QuantForge 
QuantForge is a webapp using Python’s dash framework that allows users to create custom financial trading strategies with basic stock indicators. It includes a backtesting module that allows user to test their strategies on historical data for a chosen stock and obtain key metrics about their strategy and returns. Additionally, Alpaca’s trading API is used to simulate live trading that places buy and sell orders based on the users’ custom indicator conditions, utilizing SQLite in Python to handle backend data management and store strategy info, orders, and portfolio info

### Core Features
- Custom strategy creation with customizable indicator parameters based on 6 commonly used financial indicators: RSI, MACD, SMA Crossover, SMI, Bollinger Bands, Stochastic Oscillator
- Historical backtesting with user-specified date, stock, and timeframe parameters based on the user-created strategy which displays buy and sell orders, hypothetical portfolio performance, and key backtesting statistics
- Live papertrading which accesses live market data and places paper orders based on the criterion set by the chosen strategy, displaying user portfolio returns and placed orders 
![image](https://github.com/user-attachments/assets/1892500a-e998-4215-86da-60b9c39d183d)

### Setup Instructions:
To access features that require the Alpaca API, you must place your own Alpaca API key in an environment file prior to running 


Instructions:
- Create a file in your repository directory called .env
- Create a variable "ALPACA_KEY" and set it equal to your user's alpaca key
- Create a variable "SECRET_KEY" and set it equal to your user's secret key
- Run the main "strategy.py" file to locally host the site
