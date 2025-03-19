# QuantForge 
### QuantForge is a webapp using Python’s dash framework that allows users to create custom financial trading strategies with basic stock indicators. It includes a backtesting module that allows user to test their strategies on historical data for a chosen stock and obtain key metrics about their strategy and returns. Additionally, Alpaca’s trading API is used to simulate live trading that places buy and sell orders based on the users’ custom indicator conditions, utilizing SQLite in Python to handle backend data management and store strategy info, orders, and portfolio info

### Setup Instructions:
To access features that require the Alpaca API, you must place your own Alpaca API key in an environment file prior to running 


Instructions:
- Create a file in your repository directory called .env
- Create a variable "ALPACA_KEY" and set it equal to your user's alpaca key
- Creae a variable "SECRET_KEY" and set it equal to your user's secret key
