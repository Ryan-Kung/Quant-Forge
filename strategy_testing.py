import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import sqlite3
import json
import time
import logging
from datetime import datetime, timedelta
import ta
import threading
import os
from dotenv import load_dotenv
load_dotenv()


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

API_KEY = os.getenv('ALPACA_KEY')
API_SECRET = os.getenv('SECRET_KEY')
BASE_URL = "https://paper-api.alpaca.markets"

# Initialize Alpaca API
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

# Default trading parameters
DEFAULT_ORDER_SIZE = 1
DEFAULT_TIMEFRAME = '1Min'
DEFAULT_LOOKBACK = 100  # Number of bars to fetch

class TradingStrategy:
    def __init__(self, strategy_id, name, risk_return_preference, indicators):
        self.strategy_id = strategy_id
        self.name = name
        self.risk_return_preference = float(risk_return_preference)
        self.indicators = indicators
        self.positions = {}
        self.active = True
        
    def __str__(self):
        return f"Strategy: {self.name} (ID: {self.strategy_id}), Risk-Return: {self.risk_return_preference}"

class StrategyExecutor:
    def __init__(self):
        self.strategies = {}
        self.db_path = "strategy_db.sqlite"
        self.running_threads = {}
        self.symbols = ["AAPL", "MSFT", "AMZN", "TSLA", "GOOGL"]  # Default watchlist
        
    def load_strategies_from_db(self):
        """Load all strategies from the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, risk_return_preference, indicators FROM strategy")
        
        for row in cursor.fetchall():
            strategy_id, name, risk_return_preference, indicators_json = row
            indicators = json.loads(indicators_json) if indicators_json else []
            
            # Create strategy object
            strategy = TradingStrategy(
                strategy_id=strategy_id,
                name=name,
                risk_return_preference=risk_return_preference, 
                indicators=indicators
            )
            
            self.strategies[strategy_id] = strategy
            logger.info(f"Loaded strategy: {strategy}")
            
        conn.close()
        logger.info(f"Loaded {len(self.strategies)} strategies from database")
        return self.strategies
    
    def get_market_data(self, symbol, timeframe=DEFAULT_TIMEFRAME, limit=DEFAULT_LOOKBACK):
        """Fetch market data for a symbol."""
        try:
            # Get data from Alpaca
            bars = api.get_bars(symbol, timeframe, limit=limit).df
            
            if bars.empty:
                logger.warning(f"No data returned for {symbol}")
                return None
                
            # Ensure we have all required columns
            bars = bars.rename(columns={
                'open': 'Open', 
                'high': 'High', 
                'low': 'Low', 
                'close': 'Close', 
                'volume': 'Volume'
            })
            
            return bars
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def calculate_indicators(self, df, indicator_config):
        """Calculate indicators based on configuration."""
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        for indicator in indicator_config:
            indicator_type = indicator.get("type", "")
            
            # In a real implementation, you would parse the actual settings from the database
            # For now, we'll use placeholder logic
            if indicator_type == "RSI":
                period = 14  # Default value
                result_df['RSI'] = ta.momentum.RSIIndicator(close=result_df['Close'], window=period).rsi()
                
            elif indicator_type == "MACD":
                fast_period = 12  # Default value
                slow_period = 26  # Default value
                signal_period = 9  # Default value
                
                macd = ta.trend.MACD(
                    close=result_df['Close'],
                    window_slow=slow_period,
                    window_fast=fast_period,
                    window_sign=signal_period
                )
                result_df['MACD'] = macd.macd()
                result_df['MACD_Signal'] = macd.macd_signal()
                result_df['MACD_Hist'] = macd.macd_diff()
                
            elif indicator_type == "SMA_CROSS":
                fast_period = 10  # Default value
                slow_period = 50  # Default value
                
                result_df['SMA_Fast'] = ta.trend.SMAIndicator(close=result_df['Close'], window=fast_period).sma_indicator()
                result_df['SMA_Slow'] = ta.trend.SMAIndicator(close=result_df['Close'], window=slow_period).sma_indicator()
                
            elif indicator_type == "BBANDS":
                period = 20  # Default value
                std_dev = 2  # Default value
                
                bollinger = ta.volatility.BollingerBands(
                    close=result_df['Close'],
                    window=period,
                    window_dev=std_dev
                )
                result_df['BB_Upper'] = bollinger.bollinger_hband()
                result_df['BB_Middle'] = bollinger.bollinger_mavg()
                result_df['BB_Lower'] = bollinger.bollinger_lband()
                
            elif indicator_type == "STOCH":
                k_period = 14  # Default value
                d_period = 3  # Default value
                
                stoch = ta.momentum.StochasticOscillator(
                    high=result_df['High'],
                    low=result_df['Low'],
                    close=result_df['Close'],
                    window=k_period,
                    smooth_window=d_period
                )
                result_df['STOCH_K'] = stoch.stoch()
                result_df['STOCH_D'] = stoch.stoch_signal()
                
            # Add more indicators as needed
                
        return result_df
    
    def check_buy_conditions(self, df, strategy):
        """Check if buy conditions are met."""
        # Get the latest data
        latest = df.iloc[-1]
        signals = []
        
        for indicator in strategy.indicators:
            indicator_type = indicator.get("type", "")
            
            # Check buy conditions based on indicator type
            if indicator_type == "RSI":
                # Buy when RSI is oversold
                if 'RSI' in latest and latest['RSI'] is not None:
                    oversold_level = 30  # Default value
                    if latest['RSI'] < oversold_level:
                        signals.append(True)
                    else:
                        signals.append(False)
                        
            elif indicator_type == "MACD":
                # Buy when MACD crosses above signal line
                if 'MACD' in latest and 'MACD_Signal' in latest:
                    if latest['MACD'] > latest['MACD_Signal'] and df.iloc[-2]['MACD'] <= df.iloc[-2]['MACD_Signal']:
                        signals.append(True)
                    else:
                        signals.append(False)
                        
            elif indicator_type == "SMA_CROSS":
                # Buy when fast SMA crosses above slow SMA
                if 'SMA_Fast' in latest and 'SMA_Slow' in latest:
                    if latest['SMA_Fast'] > latest['SMA_Slow'] and df.iloc[-2]['SMA_Fast'] <= df.iloc[-2]['SMA_Slow']:
                        signals.append(True)
                    else:
                        signals.append(False)
                        
            elif indicator_type == "BBANDS":
                # Buy when price crosses below lower Bollinger Band
                if 'BB_Lower' in latest:
                    if latest['Close'] < latest['BB_Lower']:
                        signals.append(True)
                    else:
                        signals.append(False)
                        
            elif indicator_type == "STOCH":
                # Buy when Stochastic K crosses above D from oversold
                if 'STOCH_K' in latest and 'STOCH_D' in latest:
                    if (latest['STOCH_K'] > latest['STOCH_D'] and 
                        df.iloc[-2]['STOCH_K'] <= df.iloc[-2]['STOCH_D'] and
                        latest['STOCH_K'] < 30):
                        signals.append(True)
                    else:
                        signals.append(False)
        
        # If we have signals and at least one is True (and none are missing)
        if signals and any(signals) and all(s is not None for s in signals):
            return True
        return False
    
    def check_sell_conditions(self, df, strategy, entry_price):
        """Check if sell conditions are met."""
        # Get the latest data
        latest = df.iloc[-1]
        current_price = latest['Close']
        signals = []
        
        # First, check risk-reward based on strategy preference
        profit_target = entry_price * (1 + 0.01 * strategy.risk_return_preference)
        stop_loss = entry_price * (1 - 0.01 * strategy.risk_return_preference)
        
        # Check if price hit profit target or stop loss
        if current_price >= profit_target or current_price <= stop_loss:
            return True
        
        # Then check sell signals from indicators
        for indicator in strategy.indicators:
            indicator_type = indicator.get("type", "")
            
            # Check sell conditions based on indicator type
            if indicator_type == "RSI":
                # Sell when RSI is overbought
                if 'RSI' in latest and latest['RSI'] is not None:
                    overbought_level = 70  # Default value
                    if latest['RSI'] > overbought_level:
                        signals.append(True)
                    else:
                        signals.append(False)
                        
            elif indicator_type == "MACD":
                # Sell when MACD crosses below signal line
                if 'MACD' in latest and 'MACD_Signal' in latest:
                    if latest['MACD'] < latest['MACD_Signal'] and df.iloc[-2]['MACD'] >= df.iloc[-2]['MACD_Signal']:
                        signals.append(True)
                    else:
                        signals.append(False)
                        
            elif indicator_type == "SMA_CROSS":
                # Sell when fast SMA crosses below slow SMA
                if 'SMA_Fast' in latest and 'SMA_Slow' in latest:
                    if latest['SMA_Fast'] < latest['SMA_Slow'] and df.iloc[-2]['SMA_Fast'] >= df.iloc[-2]['SMA_Slow']:
                        signals.append(True)
                    else:
                        signals.append(False)
                        
            elif indicator_type == "BBANDS":
                # Sell when price crosses above upper Bollinger Band
                if 'BB_Upper' in latest:
                    if latest['Close'] > latest['BB_Upper']:
                        signals.append(True)
                    else:
                        signals.append(False)
                        
            elif indicator_type == "STOCH":
                # Sell when Stochastic K crosses below D from overbought
                if 'STOCH_K' in latest and 'STOCH_D' in latest:
                    if (latest['STOCH_K'] < latest['STOCH_D'] and 
                        df.iloc[-2]['STOCH_K'] >= df.iloc[-2]['STOCH_D'] and
                        latest['STOCH_K'] > 70):
                        signals.append(True)
                    else:
                        signals.append(False)
        
        # If we have signals and at least one is True (and none are missing)
        if signals and any(signals) and all(s is not None for s in signals):
            return True
        return False
    
    def place_buy_order(self, symbol, strategy):
        """Place a buy order for a symbol using the Alpaca API."""
        try:
            # Scale order size based on risk preference
            order_size = max(1, int(DEFAULT_ORDER_SIZE * strategy.risk_return_preference))
            
            order = api.submit_order(
                symbol=symbol,
                qty=order_size,
                side='buy',
                type='market',
                time_in_force='gtc'
            )
            
            # Record the position
            entry_price = float(api.get_latest_trade(symbol).price)
            strategy.positions[symbol] = {
                'entry_price': entry_price,
                'quantity': order_size,
                'order_id': order.id,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"BUY order placed: {symbol} for strategy {strategy.name}, qty: {order_size}, price: {entry_price}")
            return True
            
        except Exception as e:
            logger.error(f"Error placing buy order for {symbol}: {e}")
            return False
    
    def place_sell_order(self, symbol, strategy):
        """Place a sell order for a symbol using the Alpaca API."""
        try:
            if symbol not in strategy.positions:
                logger.warning(f"No position found for {symbol} in strategy {strategy.name}")
                return False
                
            position = strategy.positions[symbol]
            
            order = api.submit_order(
                symbol=symbol,
                qty=position['quantity'],
                side='sell',
                type='market',
                time_in_force='gtc'
            )
            
            exit_price = float(api.get_latest_trade(symbol).price)
            profit_loss = (exit_price - position['entry_price']) * position['quantity']
            
            logger.info(f"SELL order placed: {symbol} for strategy {strategy.name}, qty: {position['quantity']}, entry: {position['entry_price']}, exit: {exit_price}, P/L: {profit_loss:.2f}")
            
            # Remove the position
            del strategy.positions[symbol]
            return True
            
        except Exception as e:
            logger.error(f"Error placing sell order for {symbol}: {e}")
            return False
    
    def execute_strategy(self, strategy, symbol):
        """Execute a strategy for a specific symbol."""
        try:
            # Check if market is open
            clock = api.get_clock()
            if not clock.is_open:
                logger.info("Market is closed. Waiting for market open.")
                return
            
            # Get market data
            df = self.get_market_data(symbol)
            if df is None or df.empty:
                logger.warning(f"No data available for {symbol}")
                return
                
            # Calculate indicators
            df = self.calculate_indicators(df, strategy.indicators)
            
            # Check for positions in this symbol for this strategy
            has_position = symbol in strategy.positions
            
            if not has_position:
                # Check buy conditions
                if self.check_buy_conditions(df, strategy):
                    self.place_buy_order(symbol, strategy)
            else:
                # Check sell conditions
                entry_price = strategy.positions[symbol]['entry_price']
                if self.check_sell_conditions(df, strategy, entry_price):
                    self.place_sell_order(symbol, strategy)
                    
        except Exception as e:
            logger.error(f"Error executing strategy {strategy.name} for {symbol}: {e}")
    
    def run_strategy_for_symbol(self, strategy, symbol):
        """Run a strategy for a symbol in a continuous loop."""
        logger.info(f"Starting execution of strategy {strategy.name} for {symbol}")
        
        while strategy.active:
            try:
                self.execute_strategy(strategy, symbol)
                # Sleep for 1 minute
                time.sleep(60)
            except Exception as e:
                logger.error(f"Error in run_strategy_for_symbol: {e}")
                time.sleep(60)  # Sleep on error to avoid rapid retries
    
    def start_strategy(self, strategy_id, symbols=None):
        """Start executing a strategy for the given symbols."""
        if strategy_id not in self.strategies:
            logger.error(f"Strategy ID {strategy_id} not found")
            return False
            
        strategy = self.strategies[strategy_id]
        symbols_to_trade = symbols if symbols else self.symbols
        
        logger.info(f"Starting strategy {strategy.name} for symbols: {symbols_to_trade}")
        
        # Create and start threads for each symbol
        for symbol in symbols_to_trade:
            thread_name = f"{strategy_id}_{symbol}"
            
            if thread_name in self.running_threads:
                logger.warning(f"Strategy already running for {symbol}")
                continue
                
            thread = threading.Thread(
                target=self.run_strategy_for_symbol,
                args=(strategy, symbol),
                name=thread_name,
                daemon=True
            )
            
            thread.start()
            self.running_threads[thread_name] = thread
            logger.info(f"Started thread for strategy {strategy.name} on {symbol}")
            
        return True
    
    def stop_strategy(self, strategy_id):
        """Stop executing a strategy."""
        if strategy_id not in self.strategies:
            logger.error(f"Strategy ID {strategy_id} not found")
            return False
            
        strategy = self.strategies[strategy_id]
        strategy.active = False
        
        # Clean up threads
        for thread_name in list(self.running_threads.keys()):
            if thread_name.startswith(f"{strategy_id}_"):
                # Thread will terminate on next loop iteration since active=False
                del self.running_threads[thread_name]
                
        logger.info(f"Stopped strategy {strategy.name}")
        return True
    
    def start_all_strategies(self):
        """Start executing all loaded strategies."""
        for strategy_id in self.strategies:
            self.start_strategy(strategy_id)
    
    def stop_all_strategies(self):
        """Stop executing all running strategies."""
        for strategy_id in list(self.strategies.keys()):
            self.stop_strategy(strategy_id)

# Example usage
if __name__ == "__main__":
    executor = StrategyExecutor()
    
    # Load strategies from database
    executor.load_strategies_from_db()
    
    # Check if we have strategies
    if not executor.strategies:
        logger.warning("No strategies found in database. Please create strategies first.")
        exit(1)
    
    try:
        # Start all strategies
        executor.start_all_strategies()
        
        # Keep main thread alive
        while True:
            time.sleep(60)
            
    except KeyboardInterrupt:
        logger.info("Stopping all strategies...")
        executor.stop_all_strategies()
        logger.info("Exiting...")