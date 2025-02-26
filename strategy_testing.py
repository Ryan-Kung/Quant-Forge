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

logging.basicConfig(
level=logging.INFO,
format='%(asctime)s - %(levelname)s - %(message)s',
handlers=[
logging.FileHandler("trading_log.txt"),
logging.StreamHandler()
]
)
logger = logging.getLogger('trading_log')

API_KEY = os.getenv('ALPACA_KEY')
API_SECRET = os.getenv('SECRET_KEY')
BASE_URL = "https://paper-api.alpaca.markets"

api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

DEFAULT_ORDER_SIZE = 1
DEFAULT_TIMEFRAME = '1Min'
DEFAULT_LOOKBACK = 100 # Number of bars to fetch per request

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
    def __init__(self, simulate_mode=False):
        self.strategies = {}
        self.db_path = "strategy_db.sqlite"
        self.running_threads = {}
        self.symbols = ["QQQ"] # Default watchlist
        self.simulate_mode = simulate_mode
        # Dictionary to hold historical data dataframes in simulation mode:
        self.historical_data = {}
# In simulation mode, we maintain a pointer (row index) for each symbol.
        self.simulation_index = {}
    def load_strategies_from_db(self):
        """Load all strategies from the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, risk_return_preference, indicators FROM strategy")
        
        for row in cursor.fetchall():
            strategy_id, name, risk_return_preference, indicators_json = row
            indicators = json.loads(indicators_json) if indicators_json else []
            
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
        """Fetch the most recent market data (live mode)."""
        try:
            bars = api.get_bars(symbol, timeframe, limit=limit).df
            if bars.empty:
                logger.warning(f"No live data returned for {symbol}")
                return None
            bars = bars.rename(columns={
                'open': 'Open', 
                'high': 'High', 
                'low': 'Low', 
                'close': 'Close', 
                'volume': 'Volume'
            })
            return bars
        except Exception as e:
            logger.error(f"Error fetching live data for {symbol}: {e}")
            return None

    def get_historical_data(self, symbol, start, end, timeframe=DEFAULT_TIMEFRAME):
        """Fetch historical data for simulation mode."""
        try:
            bars = api.get_bars(symbol, timeframe, start=start, end=end).df
            if bars.empty:
                logger.warning(f"No historical data returned for {symbol}")
                return None
            bars = bars.rename(columns={
                'open': 'Open', 
                'high': 'High', 
                'low': 'Low', 
                'close': 'Close', 
                'volume': 'Volume'
            })
            return bars.sort_index()
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return None

    def simulate_live_stream(self, symbol):
        """Return the next row of historical data (simulating a live bar)."""
        df = self.historical_data.get(symbol)
        if df is None:
            return None
        
        idx = self.simulation_index.get(symbol, 0)
        if idx >= len(df):
            logger.info(f"Simulation data exhausted for {symbol}")
            return None
        # Return data up to the current simulated "time"
        current_data = df.iloc[:idx+1].copy()
        # Advance pointer
        self.simulation_index[symbol] = idx + 1
        return current_data

    def calculate_indicators(self, df, indicator_config):
        """Calculate indicators based on configuration."""
        result_df = df.copy()
        for indicator in indicator_config:
            indicator_type = indicator.get("type", "")
            
            if indicator_type == "RSI":
                period = 14  # You can pull your custom value from indicator["settings"]
                result_df['RSI'] = ta.momentum.RSIIndicator(close=result_df['Close'], window=period).rsi()
                
            elif indicator_type == "MACD":
                fast_period = 12  
                slow_period = 26  
                signal_period = 9  
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
                fast_period = 10  
                slow_period = 50  
                result_df['SMA_Fast'] = ta.trend.SMAIndicator(close=result_df['Close'], window=fast_period).sma_indicator()
                result_df['SMA_Slow'] = ta.trend.SMAIndicator(close=result_df['Close'], window=slow_period).sma_indicator()
                
            elif indicator_type == "BBANDS":
                period = 20  
                std_dev = 2  
                bollinger = ta.volatility.BollingerBands(
                    close=result_df['Close'],
                    window=period,
                    window_dev=std_dev
                )
                result_df['BB_Upper'] = bollinger.bollinger_hband()
                result_df['BB_Middle'] = bollinger.bollinger_mavg()
                result_df['BB_Lower'] = bollinger.bollinger_lband()
                
            elif indicator_type == "STOCH":
                k_period = 14  
                d_period = 3  
                stoch = ta.momentum.StochasticOscillator(
                    high=result_df['High'],
                    low=result_df['Low'],
                    close=result_df['Close'],
                    window=k_period,
                    smooth_window=d_period
                )
                result_df['STOCH_K'] = stoch.stoch()
                result_df['STOCH_D'] = stoch.stoch_signal()
        return result_df

    def check_buy_conditions(self, df, strategy):
        """Check if buy conditions are met."""
        latest = df.iloc[-1]
        signals = []
        for indicator in strategy.indicators:
            indicator_type = indicator.get("type", "")
            if indicator_type == "RSI":
                if 'RSI' in latest and pd.notna(latest['RSI']):
                    oversold_level = 30  
                    signals.append(latest['RSI'] < oversold_level)
            elif indicator_type == "MACD":
                if 'MACD' in latest and 'MACD_Signal' in latest:
                    # A simple cross check using the previous bar too
                    if (latest['MACD'] > latest['MACD_Signal'] and 
                        df.iloc[-2]['MACD'] <= df.iloc[-2]['MACD_Signal']):
                        signals.append(True)
                    else:
                        signals.append(False)
            elif indicator_type == "SMA_CROSS":
                if 'SMA_Fast' in latest and 'SMA_Slow' in latest:
                    if (latest['SMA_Fast'] > latest['SMA_Slow'] and 
                        df.iloc[-2]['SMA_Fast'] <= df.iloc[-2]['SMA_Slow']):
                        signals.append(True)
                    else:
                        signals.append(False)
            elif indicator_type == "BBANDS":
                if 'BB_Lower' in latest:
                    signals.append(latest['Close'] < latest['BB_Lower'])
            elif indicator_type == "STOCH":
                if 'STOCH_K' in latest and 'STOCH_D' in latest:
                    condition = (latest['STOCH_K'] > latest['STOCH_D'] and 
                                df.iloc[-2]['STOCH_K'] <= df.iloc[-2]['STOCH_D'] and
                                latest['STOCH_K'] < 30)
                    signals.append(condition)
        return True if signals and any(signals) and all(s is not None for s in signals) else False

    def check_sell_conditions(self, df, strategy, entry_price):
        """Check if sell conditions are met."""
        latest = df.iloc[-1]
        current_price = latest['Close']
        profit_target = entry_price * (1 + 0.01 * strategy.risk_return_preference)
        stop_loss = entry_price * (1 - 0.01 * strategy.risk_return_preference)
        
        if current_price >= profit_target or current_price <= stop_loss:
            return True
        
        signals = []
        for indicator in strategy.indicators:
            indicator_type = indicator.get("type", "")
            if indicator_type == "RSI":
                if 'RSI' in latest and pd.notna(latest['RSI']):
                    overbought_level = 70  
                    signals.append(latest['RSI'] > overbought_level)
            elif indicator_type == "MACD":
                if 'MACD' in latest and 'MACD_Signal' in latest:
                    if (latest['MACD'] < latest['MACD_Signal'] and 
                        df.iloc[-2]['MACD'] >= df.iloc[-2]['MACD_Signal']):
                        signals.append(True)
                    else:
                        signals.append(False)
            elif indicator_type == "SMA_CROSS":
                if 'SMA_Fast' in latest and 'SMA_Slow' in latest:
                    if (latest['SMA_Fast'] < latest['SMA_Slow'] and 
                        df.iloc[-2]['SMA_Fast'] >= df.iloc[-2]['SMA_Slow']):
                        signals.append(True)
                    else:
                        signals.append(False)
            elif indicator_type == "BBANDS":
                if 'BB_Upper' in latest:
                    signals.append(latest['Close'] > latest['BB_Upper'])
            elif indicator_type == "STOCH":
                if 'STOCH_K' in latest and 'STOCH_D' in latest:
                    condition = (latest['STOCH_K'] < latest['STOCH_D'] and 
                                df.iloc[-2]['STOCH_K'] >= df.iloc[-2]['STOCH_D'] and
                                latest['STOCH_K'] > 70)
                    signals.append(condition)
        return True if signals and any(signals) and all(s is not None for s in signals) else False

    def place_buy_order(self, symbol, strategy, price=None):
        """Place (or simulate) a buy order."""
        try:
            order_size = max(1, int(DEFAULT_ORDER_SIZE * strategy.risk_return_preference))
            
            if self.simulate_mode:
                # In simulation mode simply simulate an order execution using the latest price provided
                entry_price = price if price else 0
                order_id = f"SIM_BUY_{datetime.now().timestamp()}"
            else:
                order = api.submit_order(
                    symbol=symbol,
                    qty=order_size,
                    side='buy',
                    type='market',
                    time_in_force='gtc'
                )
                entry_price = float(api.get_latest_trade(symbol).price)
                order_id = order.id
            
            strategy.positions[symbol] = {
                'entry_price': entry_price,
                'quantity': order_size,
                'order_id': order_id,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"BUY order executed for {symbol} on strategy {strategy.name}: qty={order_size}, price={entry_price}")
            return True
        except Exception as e:
            logger.error(f"Error placing buy order for {symbol}: {e}")
            return False

    def place_sell_order(self, symbol, strategy, price=None):
        """Place (or simulate) a sell order."""
        try:
            if symbol not in strategy.positions:
                logger.warning(f"No open position for {symbol} in strategy {strategy.name}")
                return False
                
            position = strategy.positions[symbol]
            
            if self.simulate_mode:
                exit_price = price if price else 0
                order_id = f"SIM_SELL_{datetime.now().timestamp()}"
            else:
                order = api.submit_order(
                    symbol=symbol,
                    qty=position['quantity'],
                    side='sell',
                    type='market',
                    time_in_force='gtc'
                )
                exit_price = float(api.get_latest_trade(symbol).price)
                order_id = order.id
            
            profit_loss = (exit_price - position['entry_price']) * position['quantity']
            logger.info(f"SELL executed for {symbol} on strategy {strategy.name}: qty={position['quantity']}, entry={position['entry_price']}, exit={exit_price}, P/L={profit_loss:.2f}")
            
            del strategy.positions[symbol]
            return True
        except Exception as e:
            logger.error(f"Error placing sell order for {symbol}: {e}")
            return False

    def execute_strategy(self, strategy, symbol, df):
        """Use the supplied data frame (simulated or live) for executing the strategy."""
        try:
            df = self.calculate_indicators(df, strategy.indicators)
            has_position = symbol in strategy.positions
            
            latest_price = df.iloc[-1]['Close']
            if not has_position:
                if self.check_buy_conditions(df, strategy):
                    self.place_buy_order(symbol, strategy, price=latest_price)
            else:
                entry_price = strategy.positions[symbol]['entry_price']
                if self.check_sell_conditions(df, strategy, entry_price):
                    self.place_sell_order(symbol, strategy, price=latest_price)
        except Exception as e:
            logger.error(f"Error executing strategy {strategy.name} for {symbol}: {e}")

    def run_strategy_for_symbol(self, strategy, symbol):
        """Run a strategy for a symbol continuously (live or simulation mode)."""
        logger.info(f"Started strategy {strategy.name} for {symbol}")
        
        while strategy.active:
            try:
                if self.simulate_mode:
                    # use simulated historical data stream
                    df = self.simulate_live_stream(symbol)
                    if df is None or df.empty:
                        logger.info(f"Simulation for {symbol} completed.")
                        break
                    # In simulation, pause briefly (or mimic the timeframe interval)
                    time.sleep(1)
                else:
                    # Check if market is open
                    clock = api.get_clock()
                    if not clock.is_open:
                        logger.info("Market is closed. Waiting for market open.")
                        time.sleep(60)
                        continue
                    df = self.get_market_data(symbol)
                    if df is None or df.empty:
                        logger.warning(f"No data available for {symbol}")
                        time.sleep(60)
                        continue
                    # In live mode, pause for a minute (or the appropriate timeframe)
                    time.sleep(60)
                    
                self.execute_strategy(strategy, symbol, df)
            except Exception as e:
                logger.error(f"Error in run_strategy_for_symbol for {symbol}: {e}")
                time.sleep(60)

    def start_strategy(self, strategy_id, symbols=None):
        """Start executing a strategy for the given symbols."""
        if strategy_id not in self.strategies:
            logger.error(f"Strategy ID {strategy_id} not found")
            return False
            
        strategy = self.strategies[strategy_id]
        symbols_to_trade = symbols if symbols else self.symbols
        
        if self.simulate_mode:
            # Preload historical data for simulation. For example, simulate data from the past week.
            end = datetime.now() - timedelta(days = 1)
            start = end - timedelta(days=7)
            for sym in symbols_to_trade:
                hist = self.get_historical_data(sym, start.isoformat()[:10], end.isoformat()[:10])
                if hist is None or hist.empty:
                    logger.error(f"No historical data for {sym}; cannot simulate.")
                else:
                    self.historical_data[sym] = hist
                    self.simulation_index[sym] = 0
                    
        logger.info(f"Starting strategy {strategy.name} for symbols: {symbols_to_trade}")
        
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
            logger.info(f"Started thread for {strategy.name} on {symbol}")
            
        return True

    def stop_strategy(self, strategy_id):
        """Stop executing a strategy."""
        if strategy_id not in self.strategies:
            logger.error(f"Strategy ID {strategy_id} not found")
            return False
            
        strategy = self.strategies[strategy_id]
        strategy.active = False
        
        for thread_name in list(self.running_threads.keys()):
            if thread_name.startswith(f"{strategy_id}_"):
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


executor = StrategyExecutor(simulate_mode=True)
executor.load_strategies_from_db()
if not executor.strategies:
    logger.warning("No strategies found in database. Please create strategies first.")
    exit(1)

try:
    executor.start_all_strategies()
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    logger.info("Stopping all strategies...")
    executor.stop_all_strategies()
    logger.info("Exiting...")