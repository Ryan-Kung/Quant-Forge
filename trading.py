import sqlite3
import json
import alpaca_trade_api as tradeapi
import pandas as pd
import time
import ta

class DatabaseManager:
    def __init__(self, db_name='trading_strategies.db'):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.create_tables()

    def create_tables(self):
        # Create tables if they don't exist
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategies (
                strategy_id INTEGER PRIMARY KEY,
                symbol TEXT NOT NULL,
                order_size REAL NOT NULL,
                is_active INTEGER DEFAULT 1
            )
        ''')

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategy_indicators (
                indicator_id INTEGER PRIMARY KEY,
                strategy_id INTEGER,
                indicator_name TEXT NOT NULL,
                parameters TEXT NOT NULL,
                FOREIGN KEY (strategy_id) REFERENCES strategies (strategy_id)
            )
        ''')

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategy_conditions (
                condition_id INTEGER PRIMARY KEY,
                indicator_id INTEGER,
                condition_type TEXT NOT NULL,
                value TEXT NOT NULL,
                comparison_operator TEXT NOT NULL,
                FOREIGN KEY (indicator_id) REFERENCES strategy_indicators (indicator_id)
            )
        ''')
        self.conn.commit()

    def add_strategy(self, symbol, order_size, indicators, conditions):
        """
        Add a new strategy with its indicators and conditions
        
        indicators format: [{'name': 'SMA', 'parameters': {'period': 10}}, ...]
        conditions format: [{'indicator_index': 0, 'type': 'value', 'value': '200', 'operator': '>'}, ...]
        """
        try:
            # Insert strategy
            self.cursor.execute(
                "INSERT INTO strategies (symbol, order_size) VALUES (?, ?)",
                (symbol, order_size)
            )
            strategy_id = self.cursor.lastrowid

            # Insert indicators
            for ind in indicators:
                self.cursor.execute(
                    "INSERT INTO strategy_indicators (strategy_id, indicator_name, parameters) VALUES (?, ?, ?)",
                    (strategy_id, ind['name'], json.dumps(ind['parameters']))
                )
                indicator_id = self.cursor.lastrowid

                # Insert conditions for this indicator
                for cond in conditions:
                    if cond['indicator_index'] == indicators.index(ind):
                        self.cursor.execute(
                            """INSERT INTO strategy_conditions 
                               (indicator_id, condition_type, value, comparison_operator) 
                               VALUES (?, ?, ?, ?)""",
                            (indicator_id, cond['type'], cond['value'], cond['operator'])
                        )

            self.conn.commit()
            return strategy_id
        except Exception as e:
            print(f"Error adding strategy: {e}")
            self.conn.rollback()
            return None

    def get_all_active_strategies(self):
        """Get all active strategies with their indicators and conditions"""
        self.cursor.execute("""
            SELECT 
                s.strategy_id,
                s.symbol,
                s.order_size,
                i.indicator_name,
                i.parameters,
                c.condition_type,
                c.value,
                c.comparison_operator
            FROM strategies s
            JOIN strategy_indicators i ON s.strategy_id = i.strategy_id
            JOIN strategy_conditions c ON i.indicator_id = c.indicator_id
            WHERE s.is_active = 1
        """)
        return self.cursor.fetchall()

class TradingSystem:
    def __init__(self, api_key, secret_key, base_url):
        self.api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
        self.db = DatabaseManager()

    def process_strategies(self, strategies_data):
        """Convert database records into structured strategy objects"""
        processed_strategies = {}
        
        for row in strategies_data:
            (strategy_id, symbol, order_size, indicator_name, 
             parameters, condition_type, value, operator) = row
            
            if strategy_id not in processed_strategies:
                processed_strategies[strategy_id] = {
                    'symbol': symbol,
                    'order_size': order_size,
                    'indicators': {},
                    'conditions': []
                }
            
            # Add indicator
            processed_strategies[strategy_id]['indicators'][indicator_name] = json.loads(parameters)
            
            # Add condition
            processed_strategies[strategy_id]['conditions'].append({
                'indicator': indicator_name,
                'type': condition_type,
                'value': value,
                'operator': operator
            })
        
        return processed_strategies

    def calculate_indicators(self, df, indicators):
        """Calculate all required indicators for a strategy"""
        results = {}
        for indicator_name, params in indicators.items():
            if indicator_name == 'SMA':
                results[indicator_name] = df['close'].rolling(window=params['period']).mean()
            elif indicator_name == 'RSI':
                results[indicator_name] = ta.momentum.RSIIndicator(
                    df['close'], window=params['period']
                ).rsi()
            # Add more indicators as needed
        return results

    def evaluate_conditions(self, conditions, indicator_values):
        """Evaluate all conditions for a strategy"""
        for condition in conditions:
            operator = condition['operator']
            indicator_value = indicator_values[condition['indicator']].iloc[-1]
            comparison_value = float(condition['value'])
            
            if not self.compare_values(indicator_value, comparison_value, operator):
                return False
        return True

    @staticmethod
    def compare_values(a, b, operator):
        """Compare two values based on the operator"""
        operators = {
            '>': lambda x, y: x > y,
            '<': lambda x, y: x < y,
            '>=': lambda x, y: x >= y,
            '<=': lambda x, y: x <= y,
            '==': lambda x, y: x == y
        }
        return operators[operator](a, b)

    def place_order(self, symbol, side, qty):
        """Place an order through Alpaca"""
        try:
            self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type='market',
                time_in_force='gtc'
            )
            print(f"Order placed: {side} {qty} shares of {symbol}")
        except Exception as e:
            print(f"Order failed: {e}")

    def run(self):
        """Main trading loop"""
        while True:
            try:
                # Get all active strategies
                strategies_data = self.db.get_all_active_strategies()
                strategies = self.process_strategies(strategies_data)

                # Process each strategy
                for strategy_id, strategy in strategies.items():
                    symbol = strategy['symbol']
                    
                    # Get latest data
                    df = self.api.get_bars(
                        symbol,
                        timeframe='1Min',
                        limit=100
                    ).df
                    
                    # Calculate indicators
                    indicator_values = self.calculate_indicators(df, strategy['indicators'])
                    
                    # Check if conditions are met
                    if self.evaluate_conditions(strategy['conditions'], indicator_values):
                        try:
                            position = self.api.get_position(symbol)
                            # Have position, consider selling
                            self.place_order(symbol, 'sell', strategy['order_size'])
                        except:
                            # No position, consider buying
                            self.place_order(symbol, 'buy', strategy['order_size'])

                time.sleep(60)  # Check every minute

            except Exception as e:
                print(f"Error in main loop: {e}")
                time.sleep(60)

# Example usage
if __name__ == "__main__":
    # Example of adding a strategy
    db = DatabaseManager()
    
    # Example strategy: SMA crossover with RSI
    strategy_data = {
        'symbol': 'TSLA',
        'order_size': 1,
        'indicators': [
            {'name': 'SMA', 'parameters': {'period': 10}},
            {'name': 'RSI', 'parameters': {'period': 14}}
        ],
        'conditions': [
            {'indicator_index': 0, 'type': 'value', 'value': '200', 'operator': '>'},
            {'indicator_index': 1, 'type': 'value', 'value': '30', 'operator': '<'}
        ]
    }
    
    # Add the strategy to the database
    db.add_strategy(
        strategy_data['symbol'],
        strategy_data['order_size'],
        strategy_data['indicators'],
        strategy_data['conditions']
    )

    # Initialize and run trading system
    trading_system = TradingSystem(
        api_key='YOUR_API_KEY',
        secret_key='YOUR_SECRET_KEY',
        base_url="https://paper-api.alpaca.markets"
    )
    trading_system.run()