import dash
from dash import dcc, html, Input, Output, State, callback, ALL, MATCH
import dash_bootstrap_components as dbc
import uuid
import sqlite3
import json
import alpaca_trade_api as tradeapi
import pandas as pd
import pandas_ta as ta
import time
import os
import numpy as np
from dotenv import load_dotenv
import logging 
load_dotenv()

trading_threads = {}

default_key = os.getenv('ALPACA_KEY')
default_secret_key = os.getenv('SECRET_KEY')

BASE_URL = "https://paper-api.alpaca.markets"
use_api = tradeapi.REST(default_key, default_secret_key, BASE_URL, api_version='v2')

#Logging 

def setup_logging():
    """Configure logging for the trading application"""
    # Create logger
    logger = logging.getLogger('trading')
    logger.setLevel(logging.DEBUG)
    
    # Create file handler for all logs
    file_handler = logging.FileHandler('trading.log')
    file_handler.setLevel(logging.DEBUG)
    
    # Create console handler with a higher log level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatters and add to handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Initialize logger
logger = setup_logging()

# Trading Functions

def get_data(symbol, api, limit=3630):
    api = api
    barset = api.get_bars(symbol, timeframe='1Min', limit=limit).df
    df = barset[['close']].copy()
    df['close'] = df['close'].astype(float)
    return df

def get_historical_data(symbol, start, end, api):
    try:
        barset = api.get_bars(symbol, timeframe='1Min', start=start, end=end).df
        if barset.empty:
            logger.warning("No data fetched for the given symbol and date range.")
            return pd.DataFrame(columns=['open', 'high', 'low', 'close'])  # Return DataFrame with expected columns
        
        # Keep all necessary columns
        df = barset[['open', 'high', 'low', 'close']].copy()
        return df
    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")
        return pd.DataFrame(columns=['open', 'high', 'low', 'close'])
# Function to place orders
def place_order(side, symbol, qty, api):
    try:
        api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type='market',
            time_in_force='gtc'
        )
        print(f"{side.upper()} order placed for {symbol}")
    except Exception as e:
        print(f"Order failed: {e}")

def extract_indicators(strategy):
    indicators = strategy[0][3]
    indicators = json.loads(indicators)
    sma_settings = []
    rsi_settings = []
    stoch_settings = []
    smi_settings = []
    macd_settings = []
    bbands_settings = []
    risk_return = strategy[0][2]

    for indicator in indicators:
        indicator_type = indicator['type']
        indicator_settings = indicator['settings']

        if indicator_type == 'SMA':
            period = indicator_settings['period']
            sma_settings.append(period)

        elif indicator_type == 'RSI':
            period = indicator_settings['period']
            overbought = indicator_settings['overbought']
            oversold = indicator_settings['oversold']
            rsi_settings.append(period)
            rsi_settings.append(overbought)
            rsi_settings.append(oversold)

        elif indicator_type == 'STOCH':
            k_period = indicator_settings['k_period']
            d_period = indicator_settings['d_period']
            stoch_settings.append(k_period)
            stoch_settings.append(d_period)

        elif indicator_type == 'SMI':
            period = indicator_settings['period']
            signal_period = indicator_settings['signal_period']
            smi_settings.append(period)
            smi_settings.append(signal_period)

        elif indicator_type == 'MACD':
            fast_period = indicator_settings['fast_period']
            slow_period = indicator_settings['slow_period']
            signal_period = indicator_settings['signal_period']
            macd_settings.append(fast_period)
            macd_settings.append(slow_period)
            macd_settings.append(signal_period)

        elif indicator_type == 'BBANDS':
            period = indicator_settings['period']
            std_dev = indicator_settings['std_dev']
            bbands_settings.append(period)
            bbands_settings.append(std_dev)

        else:
            print("Indicator not recognized")
    return sma_settings, rsi_settings, stoch_settings, smi_settings, macd_settings, bbands_settings, risk_return

def calculate_indicators(df, sma_settings, rsi_settings, stoch_settings, smi_settings, macd_settings, bbands_settings):
    if df.empty:
        print("Empty DataFrame, skipping indicator calculation.")
        return df
        
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # SMA calculation
    if sma_settings and len(sma_settings) >= 2:
        result_df['SMA_fast'] = ta.sma(result_df['close'], length=sma_settings[0])
        result_df['SMA_slow'] = ta.sma(result_df['close'], length=sma_settings[1])
    
    # RSI calculation
    if rsi_settings and len(rsi_settings) >= 1:
        result_df['RSI'] = ta.rsi(result_df['close'], length=rsi_settings[0])
        #Log RSI calculation 
        logger.info(f"RSI calculated: {result_df['RSI'].iloc[-1]}")
    
    # Stochastic calculation
    if stoch_settings and len(stoch_settings) >= 2:
        stoch_df = ta.stoch(
            result_df['high'], 
            result_df['low'], 
            result_df['close'], 
            k=stoch_settings[0], 
            d=stoch_settings[1], 
            smooth_k=stoch_settings[2] if len(stoch_settings) > 2 else 3
        )
        # Add stochastic columns to the main dataframe
        for col in stoch_df.columns:
            result_df[col] = stoch_df[col]
    
    # SMI calculation
    if smi_settings and len(smi_settings) >= 2:
        smi_df = ta.smi(
            result_df['close'], 
            length=smi_settings[0], 
            signal=smi_settings[1]
        )
        # Add SMI columns to the main dataframe
        for col in smi_df.columns:
            result_df[col] = smi_df[col]
    
    # MACD calculation
    if macd_settings and len(macd_settings) >= 3:
        fast, slow, signal = macd_settings
        macd_df = ta.macd(
            result_df['close'], 
            fast=fast, 
            slow=slow, 
            signal=signal
        )
        # Add MACD columns to the main dataframe
        for col in macd_df.columns:
            result_df[col] = macd_df[col]
            
        # Store column names for easier reference
        result_df['macd_line_col'] = macd_df.columns[0]
        result_df['macd_signal_col'] = macd_df.columns[1]
        result_df['macd_hist_col'] = macd_df.columns[2] if len(macd_df.columns) > 2 else None
    
    # Bollinger Bands calculation
    if bbands_settings and len(bbands_settings) >= 2:
        bbands_df = ta.bbands(
            result_df['close'], 
            length=bbands_settings[0], 
            std=bbands_settings[1]
        )
        # Add BBands columns to the main dataframe
        for col in bbands_df.columns:
            result_df[col] = bbands_df[col]
            
        # Store column names for easier reference
        result_df['bbands_upper_col'] = bbands_df.columns[0]
        result_df['bbands_mid_col'] = bbands_df.columns[1]
        result_df['bbands_lower_col'] = bbands_df.columns[2]
    
    return result_df

# Check buy conditions for a SINGLE row of data
def check_buy_conditions_for_row(row, sma_settings, rsi_settings, stoch_settings, smi_settings, macd_settings, bbands_settings):
 
    conditions_met = []
    conditions_total = 0
    
    # SMA condition: fast > slow
    if sma_settings and len(sma_settings) >= 2:
        conditions_total += 1
        if 'SMA_fast' in row and 'SMA_slow' in row:
            sma_condition = row['SMA_fast'] > row['SMA_slow']
            conditions_met.append(sma_condition)
    
    # RSI condition: oversold (RSI < threshold)
    if rsi_settings and len(rsi_settings) >= 3:
        conditions_total += 1
        if 'RSI' in row:
            rsi_condition = row['RSI'] < rsi_settings[2]
            conditions_met.append(rsi_condition)
    
    # Stochastic condition
    if stoch_settings and len(stoch_settings) >= 2:
        conditions_total += 1
        k_period = stoch_settings[0]
        d_period = stoch_settings[1]
        smooth_k = stoch_settings[2] if len(stoch_settings) > 2 else 3
        
        # Typical column names
        k_col = f"STOCHk_{k_period}_{d_period}_{smooth_k}"
        d_col = f"STOCHd_{k_period}_{d_period}_{smooth_k}"
        
        if k_col in row and d_col in row:
            stoch_condition = (row[k_col] > row[d_col]) and (row[k_col] < 20) and (row[d_col] < 20)
            conditions_met.append(stoch_condition)
    
    # SMI condition
    if smi_settings and len(smi_settings) >= 2:
        conditions_total += 1
        length = smi_settings[0]
        signal_length = smi_settings[1]
        
        # Typical column names
        smi_col = f"SMI_{length}_{signal_length}"
        smi_signal_col = f"SMIs_{length}_{signal_length}"
        
        if smi_col in row and smi_signal_col in row:
            smi_condition = row[smi_col] > 0
            conditions_met.append(smi_condition)
    
    # MACD condition: MACD line > Signal line
    if macd_settings and len(macd_settings) >= 3:
        conditions_total += 1
        
        if 'macd_line_col' in row and 'macd_signal_col' in row:
            macd_col = row['macd_line_col']
            signal_col = row['macd_signal_col']
            
            if macd_col in row and signal_col in row:
                macd_condition = row[macd_col] > row[signal_col]
                conditions_met.append(macd_condition)

    
    # Bollinger Bands condition: price < lower band
    if bbands_settings and len(bbands_settings) >= 2:
        conditions_total += 1
        
        if 'bbands_lower_col' in row:
            lower_band_col = row['bbands_lower_col']
            
            if lower_band_col in row:
                bbands_condition = row['close'] < row[lower_band_col]
                conditions_met.append(bbands_condition)
                    
    # Check if we have any conditions to evaluate
    if conditions_total == 0:
        return False
    
    if len(conditions_met) == 0:
        return False
    
    # All conditions must be TRUE for a buy signal
    all_conditions_met = all(conditions_met)
    
    return all_conditions_met

def check_sell_conditions_for_row(row, entry_price, sma_settings, rsi_settings, stoch_settings, smi_settings, macd_settings, bbands_settings, risk_return_preference):
    """
    Check sell conditions for a single row of data
    Returns True if any sell condition is met
    """
    conditions_met = []
    
    # 1. Risk-Return Based Exit
    if entry_price:
        current_return = (row['close'] - entry_price) / entry_price * 100
        # Dynamic take profit and stop loss based on risk_return_preference
        take_profit = 1.0 * risk_return_preference  # Higher risk tolerance = higher profit target
        stop_loss = -0.5 * risk_return_preference   # Higher risk tolerance = wider stop loss
        
        if current_return >= take_profit or current_return <= stop_loss:
            logger.info(f"Risk-Return Exit: {current_return:.2f}%")
            return True
    
    # 2. Technical Indicator Based Exit
    
    # SMA condition: fast < slow (trend reversal)
    if sma_settings and len(sma_settings) >= 2:
        if 'SMA_fast' in row and 'SMA_slow' in row:
            sma_condition = row['SMA_fast'] < row['SMA_slow']
            conditions_met.append(sma_condition)
    
    # RSI condition: overbought
    if rsi_settings and len(rsi_settings) >= 3:
        if 'RSI' in row:
            rsi_condition = row['RSI'] > rsi_settings[1]  # Overbought level
            logger.info('RSI: ' + str(row['RSI']))
            conditions_met.append(rsi_condition)
    
    # MACD condition: bearish crossover
    if macd_settings and len(macd_settings) >= 3:
        if 'MACD' in row and 'MACD_signal' in row:
            macd_condition = row['MACD'] < row['MACD_signal']
            conditions_met.append(macd_condition)
    
    # Bollinger Bands condition: price above upper band
    if bbands_settings and len(bbands_settings) >= 2:
        if 'BB_upper' in row:
            bb_condition = row['close'] > row['BB_upper']
            logger.info('BB_upper: ' + str(row['BB_upper']))
            conditions_met.append(bb_condition)

    # Stochastic condition: k > d
    if stoch_settings and len(stoch_settings) >= 2:
        k_period = stoch_settings[0]
        d_period = stoch_settings[1]
        smooth_k = stoch_settings[2] if len(stoch_settings) > 2 else 3
        
        # Typical column names
        k_col = f"STOCHk_{k_period}_{d_period}_{smooth_k}"
        d_col = f"STOCHd_{k_period}_{d_period}_{smooth_k}"
        
        if k_col in row and d_col in row:
            stoch_condition = row[k_col] > row[d_col]
            conditions_met.append(stoch_condition)
    
    # SMI condition: SMI < 0
    if smi_settings and len(smi_settings) >= 2:
        length = smi_settings[0]
        signal_length = smi_settings[1]
        
        # Typical column names
        smi_col = f"SMI_{length}_{signal_length}"
        
        if smi_col in row:
            smi_condition = row[smi_col] < 0
            conditions_met.append(smi_condition)
    
    # Return True if all technical conditions are met
    return all(conditions_met)

def run_live_trading(symbol, strategy_id, alpaca_key, secret_key, live=True):
    # Create a unique thread ID
    thread_id = f"{strategy_id}_{symbol}"
    trading_threads[thread_id] = {"stop": False}
    
    try:
        # Initialize API with the provided credentials
        api = tradeapi.REST(alpaca_key, secret_key, BASE_URL, api_version='v2')
        logger.info(f"Alpaca API connection established for {symbol} using strategy {strategy_id}")
        
        # Rest of your function remains the same...
        db = get_strategy_db()
        cursor = db.cursor()
        cursor.execute("SELECT * FROM strategy WHERE id = ?", (strategy_id,))
        strategy = cursor.fetchall()
        
        if not strategy:
            logger.error(f"Strategy with ID {strategy_id} not found")
            return
        
        sma_settings, rsi_settings, stoch_settings, smi_settings, macd_settings, bbands_settings, risk_return_preference = extract_indicators(strategy)
        
        position = None
        entry_price = None

        i = 1
        
        while not trading_threads[thread_id]["stop"]:  # Check stop flag
            try:
                logger.debug("Fetching market data...")
                
                if live: 
                    df = get_data(symbol, api=api)
                else:
                    df = get_historical_data(symbol, start='2020-01-08', end='2020-01-09', api=api).iloc[0:i]
                    logger.info(len(df))
                if df.empty:
                    logger.warning("No data available")
                    time.sleep(60)
                    continue
                
                # Calculate indicators
                df = calculate_indicators(df, sma_settings, rsi_settings, stoch_settings, 
                                      smi_settings, macd_settings, bbands_settings)
                
                current_row = df.iloc[-1]
                
                # Check current position
                try:
                    position = api.get_position(symbol)
                    entry_price = float(position.avg_entry_price)
                except Exception as e:
                    position = None
                    entry_price = None
                
                if position is None:
                    if check_buy_conditions_for_row(current_row, sma_settings, rsi_settings, 
                                                  stoch_settings, smi_settings, macd_settings, 
                                                  bbands_settings):
                        place_order('buy', symbol, 1, api)
                        logger.info(f"Buy order placed at {current_row['close']}")
                else:
                    if check_sell_conditions_for_row(current_row, entry_price, sma_settings, 
                                                  rsi_settings, stoch_settings, smi_settings, 
                                                  macd_settings, bbands_settings, 
                                                  risk_return_preference):
                        place_order('sell', symbol, 1, api)
                        logger.info(f"Sell order placed at {current_row['close']}")
                i+=1
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in trading loop: {str(e)}")
                if trading_threads[thread_id]["stop"]:
                    break
                i+=1
                time.sleep(60)
                
    except Exception as e:
        logger.error(f"Critical error in trading process: {str(e)}", exc_info=True)
    
    # Clean up when thread exits
    logger.info(f"Trading stopped for {symbol} using strategy {strategy_id}")
    if thread_id in trading_threads:
        del trading_threads[thread_id]

# Dynamically create a backtesting Strategy class based on a strategy configuration

def create_custom_strategy_class(strategy_config):
    
    class CustomStrategy(Strategy):        
        def init(self):
            """Initialize indicators based on strategy configuration"""
            # Extract strategy parameters
            self.indicators = {}
            indicators_config = strategy_config.get("indicators", [])
            self.risk_return = float(strategy_config.get("risk_return_preference", 1.0))
            
            # Calculate each indicator based on type
            for indicator in indicators_config:
                indicator_type = indicator.get("type")
                settings = indicator.get("settings", {})
                
                if indicator_type == "SMA_CROSS":
                    fast_period = int(settings.get("fast_period", 10))
                    slow_period = int(settings.get("slow_period", 20))
                    self.indicators["sma_fast"] = self.I(SMA, self.data.Close, fast_period)
                    self.indicators["sma_slow"] = self.I(SMA, self.data.Close, slow_period)
                    
                elif indicator_type == "RSI":
                    period = int(settings.get("period", 14))
                    self.indicators["rsi"] = self.I(lambda x: self._rsi(x, period), self.data.Close)
                    self.indicators["rsi_ob"] = float(settings.get("overbought", 70))
                    self.indicators["rsi_os"] = float(settings.get("oversold", 30))
                
                elif indicator_type == "BBANDS":
                    period = int(settings.get("period", 20))
                    std_dev = float(settings.get("std_dev", 2.0))
                    # Calculate using SMA and standard deviation
                    sma = self.I(SMA, self.data.Close, period)
                    std = self.I(lambda x: self._std(x, period), self.data.Close)
                    self.indicators["bb_upper"] = self.I(lambda x, y: x + y * std_dev, sma, std)
                    self.indicators["bb_lower"] = self.I(lambda x, y: x - y * std_dev, sma, std)
                    self.indicators["bb_middle"] = sma
                
                elif indicator_type == "STOCH":
                    k_period = int(settings.get("k_period", 14))
                    d_period = int(settings.get("d_period", 3))
                    smooth_k = int(settings.get("smooth_k", 3))
                    self.indicators["stoch_k"], self.indicators["stoch_d"] = self.I(ta.stoch, self.data.High, self.data.Low, self.data.Close, k_period, d_period, smooth_k)

                elif indicator_type == "SMI":
                    period = int(settings.get("period", 13))
                    signal_period = int(settings.get("signal_period", 3))
                    self.indicators["smi"], self.indicators["smi_signal"] = self.I(ta.smi, self.data.Close, period, signal_period)

                elif indicator_type == "MACD":
                    fast_period = int(settings.get("fast_period", 12))
                    slow_period = int(settings.get("slow_period", 26))
                    signal_period = int(settings.get("signal_period", 9))
                    self.indicators["macd_line"], self.indicators["macd_signal"], self.indicators["macd_hist"] = self.I(ta.macd, self.data.Close, fast_period, slow_period, signal_period)

        
        def next(self):
            """Execute trading logic based on indicator values"""
            # Check if we should buy
            if not self.position:
                if self._check_buy_conditions():
                    self.buy()
            
            # Check if we should sell
            elif self.position:
                if self._check_sell_conditions():
                    self.position.close()
        
        def _check_buy_conditions(self):
            """Check if conditions are met for buying"""
            conditions = []
            
            # SMA crossover
            if "sma_fast" in self.indicators and "sma_slow" in self.indicators:
                # Check if fast SMA crossed above slow SMA
                if (self.indicators["sma_fast"][-1] > self.indicators["sma_slow"][-1] and 
                    self.indicators["sma_fast"][-2] <= self.indicators["sma_slow"][-2]):
                    conditions.append(True)
            
            # RSI oversold condition
            if "rsi" in self.indicators and "rsi_os" in self.indicators:
                if self.indicators["rsi"][-1] < self.indicators["rsi_os"]:
                    conditions.append(True)
            
            # Bollinger Bands lower band touch
            if "bb_lower" in self.indicators:
                if self.data.Close[-1] < self.indicators["bb_lower"][-1]:
                    conditions.append(True)

            # SMI oversold condition
            if "smi" in self.indicators and "smi_os" in self.indicators:
                if self.indicators["smi"][-1] < self.indicators["smi_os"]:
                    conditions.append(True)
            
            # MACD buy signal 
            if "macd_line" in self.indicators and "macd_signal" in self.indicators:
                if self.indicators["macd_line"][-1] > self.indicators["macd_signal"][-1]:
                    conditions.append(True)

            # Stochastic oversold condition
            if "stoch_k" in self.indicators and "stoch_d" in self.indicators:
                if (self.indicators["stoch_k"][-1] < 20 and 
                    self.indicators["stoch_d"][-1] < 20):
                    conditions.append(True)
            
            # Return True if any buy condition is met
            return any(conditions) if conditions else False
        
        def _check_sell_conditions(self):
            """Check if conditions are met for selling"""
            conditions = []
            
            # SMA crossover (bearish)
            if "sma_fast" in self.indicators and "sma_slow" in self.indicators:
                # Check if fast SMA crossed below slow SMA
                if (self.indicators["sma_fast"][-1] < self.indicators["sma_slow"][-1] and 
                    self.indicators["sma_fast"][-2] >= self.indicators["sma_slow"][-2]):
                    conditions.append(True)
            
            # RSI overbought condition
            if "rsi" in self.indicators and "rsi_ob" in self.indicators:
                if self.indicators["rsi"][-1] > self.indicators["rsi_ob"]:
                    conditions.append(True)
            
            # Bollinger Bands upper band touch
            if "bb_upper" in self.indicators:
                if self.data.Close[-1] > self.indicators["bb_upper"][-1]:
                    conditions.append(True)
            
            # SMI overbought condition
            if "smi" in self.indicators and "smi_ob" in self.indicators:
                if self.indicators["smi"][-1] > self.indicators["smi_ob"]:
                    conditions.append(True)
            
            # MACD sell signal
            if "macd_line" in self.indicators and "macd_signal" in self.indicators:
                if self.indicators["macd_line"][-1] < self.indicators["macd_signal"][-1]:
                    conditions.append(True)
            
            # Stochastic overbought condition
            if "stoch_k" in self.indicators and "stoch_d" in self.indicators:
                if (self.indicators["stoch_k"][-1] > 80 and 
                    self.indicators["stoch_d"][-1] > 80):
                    conditions.append(True)
                    
            # Risk-Return based exit
            if self.position:
                entry_price = self.position.entry_price
                current_price = self.data.Close[-1]
                pct_change = (current_price - entry_price) / entry_price * 100
                
                # Take profit or stop loss based on risk preference
                take_profit_threshold = 1.0 * self.risk_return
                stop_loss_threshold = -0.5 * self.risk_return
                
                if pct_change >= take_profit_threshold or pct_change <= stop_loss_threshold:
                    conditions.append(True)
            
            # Return True if any sell condition is met
            return any(conditions) if conditions else False
        
        # Helper methods for indicator calculations
        def _rsi(self, prices, period=14):
            """Calculate RSI for backtesting - correctly handles array inputs"""
            # Handle the case of insufficient data
            if len(prices) <= period:
                # Return an array of neutral values of the same length as prices
                return np.full_like(prices, 50.0)
                
            # Calculate price changes using numpy
            deltas = np.zeros_like(prices)
            deltas[1:] = np.diff(prices)
            
            # Separate gains and losses
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            # Initialize arrays for average gains and losses
            avg_gains = np.zeros_like(prices)
            avg_losses = np.zeros_like(prices)
            
            # Calculate first average (simple average)
            avg_gains[period] = np.mean(gains[1:period+1])
            avg_losses[period] = np.mean(losses[1:period+1])
            
            # Calculate subsequent values (using smoothing)
            for i in range(period + 1, len(prices)):
                avg_gains[i] = (avg_gains[i-1] * (period-1) + gains[i]) / period
                avg_losses[i] = (avg_losses[i-1] * (period-1) + losses[i]) / period
            
            # Calculate RS and RSI
            rs = np.zeros_like(prices)
            # Avoid division by zero
            nonzero_indices = avg_losses > 0
            rs[nonzero_indices] = avg_gains[nonzero_indices] / avg_losses[nonzero_indices]
            rs[~nonzero_indices] = 100.0
            
            # Calculate RSI
            rsi = 100.0 - (100.0 / (1.0 + rs))
            
            # Set RSI to 50 (neutral) for the starting period where we don't have enough data
            rsi[:period] = 50.0
            
            return rsi
    
    return CustomStrategy

def generate_backtest_plot(data, stats, strategy_config):
    """Generate a custom backtest plot with proper data cleaning"""
    # Create a Bokeh figure for the price chart
    p = figure(
        title=f"{strategy_config['name'] or 'Strategy'} Backtest",
        x_axis_type="datetime",
        width=1000,
        height=500,
        tools="pan,wheel_zoom,box_zoom,reset,save",
        toolbar_location="right"
    )
    
    # Create a ColumnDataSource for the data with proper NaN handling
    source_data = {
        'date': data.index,
        'open': data['Open'].replace([np.inf, -np.inf], np.nan).fillna(0).values,
        'high': data['High'].replace([np.inf, -np.inf], np.nan).fillna(0).values,
        'low': data['Low'].replace([np.inf, -np.inf], np.nan).fillna(0).values,
        'close': data['Close'].replace([np.inf, -np.inf], np.nan).fillna(0).values
    }
    
    # Initialize indicator_lines list
    indicator_lines = []
    indicators_config = strategy_config.get("indicators", [])
    
    # Calculate and add all indicators to the plot with proper error handling
    for indicator in indicators_config:
        try:
            indicator_type = indicator.get("type")
            settings = indicator.get("settings", {})
            
            if indicator_type == "SMA_CROSS":
                fast_period = int(settings.get("fast_period", 10))
                slow_period = int(settings.get("slow_period", 20))
                source_data[f'sma_{fast_period}'] = ta.sma(data['Close'], length=fast_period).fillna(0).values
                source_data[f'sma_{slow_period}'] = ta.sma(data['Close'], length=slow_period).fillna(0).values
                indicator_lines.append((f'sma_{fast_period}', f'SMA({fast_period})', '#FF7F0E'))
                indicator_lines.append((f'sma_{slow_period}', f'SMA({slow_period})', '#2CA02C'))
                
            elif indicator_type == "RSI":
                period = int(settings.get("period", 14))
                rsi_values = ta.rsi(data['Close'], length=period).fillna(50).values
                source_data['rsi'] = rsi_values
                indicator_lines.append(('rsi', f'RSI({period})', '#D62728'))
                
            elif indicator_type == "MACD":
                fast_period = int(settings.get("fast_period", 12))
                slow_period = int(settings.get("slow_period", 26))
                signal_period = int(settings.get("signal_period", 9))
                macd_data = ta.macd(data['Close'], fast=fast_period, slow=slow_period, signal=signal_period)
                for col in macd_data.columns:
                    source_data[col] = macd_data[col].fillna(0).values
                indicator_lines.append((macd_data.columns[0], f'MACD Line', '#9467BD'))
                indicator_lines.append((macd_data.columns[1], f'MACD Signal', '#8C564B'))
                
            elif indicator_type == "BBANDS":
                period = int(settings.get("period", 20))
                std_dev = float(settings.get("std_dev", 2.0))
                bbands_data = ta.bbands(data['Close'], length=period, std=std_dev)
                for col in bbands_data.columns:
                    source_data[col] = bbands_data[col].fillna(0).values
                indicator_lines.append((bbands_data.columns[0], f'BB Upper', '#FF7F0E'))
                indicator_lines.append((bbands_data.columns[1], f'BB Middle', '#2CA02C'))
                indicator_lines.append((bbands_data.columns[2], f'BB Lower', '#17BECF'))
                
            elif indicator_type == "STOCH":
                k_period = int(settings.get("k_period", 14))
                d_period = int(settings.get("d_period", 3))
                smooth_k = int(settings.get("smooth_k", 3))
                stoch_data = ta.stoch(data['High'], data['Low'], data['Close'], k=k_period, d=d_period, smooth_k=smooth_k)
                for col in stoch_data.columns:
                    source_data[col] = stoch_data[col].fillna(50).values
                indicator_lines.append((stoch_data.columns[0], f'Stochastic %K', '#E377C2'))
                indicator_lines.append((stoch_data.columns[1], f'Stochastic %D', '#7F7F7F'))
                
            elif indicator_type == "SMI":
                length = int(settings.get("period", 13))
                signal_length = int(settings.get("signal_period", 3))
                smi_data = ta.smi(data['Close'], length=length, signal=signal_length)
                for col in smi_data.columns:
                    source_data[col] = smi_data[col].fillna(0).values
                indicator_lines.append((smi_data.columns[0], f'SMI', '#BCBD22'))
                indicator_lines.append((smi_data.columns[1], f'SMI Signal', '#17BECF'))
        except Exception as e:
            print(f"Error calculating {indicator_type}: {str(e)}")
            continue
    
    # Verify all arrays are the same length
    length_check = None
    for k, v in source_data.items():
        if length_check is None:
            length_check = len(v)
        elif len(v) != length_check:
            print(f"Column {k} has length {len(v)}, expected {length_check}")
            # Pad or truncate to match length
            if len(v) > length_check:
                source_data[k] = v[:length_check]
            else:
                source_data[k] = np.pad(v, (0, length_check - len(v)), 'constant', constant_values=0)
    
    # Create the ColumnDataSource
    source = ColumnDataSource(data=source_data)
    
    # Rest of the function remains unchanged...
    # Plot price, indicators, etc.
    price_line = p.line('date', 'close', source=source, color='#1F77B4', line_width=2, legend_label="Price")
    
    # Plot indicators
    for col, label, color in indicator_lines:
        if col in source_data:  # Only plot if the column exists
            p.line('date', col, source=source, color=color, line_width=1.5, legend_label=label)
    
    # Add buy/sell markers if trades exist
    trades = stats['_trades']
    if not trades.empty:
        buy_signals = trades[trades.Size > 0]
        sell_signals = trades[trades.Size < 0]
        
        # Plot buy signals
        if not buy_signals.empty:
            p.circle(
                x=buy_signals.EntryTime,
                y=buy_signals.EntryPrice,
                size=10,
                color='green',
                alpha=0.7,
                legend_label="Buy"
            )
        
        # Plot sell signals
        if not sell_signals.empty:
            p.circle(
                x=sell_signals.EntryTime,
                y=sell_signals.ExitPrice,  # Use exit price for sell signals
                size=10,
                color='red',
                alpha=0.7,
                legend_label="Sell"
            )
    
    # Add hover tool
    hover = HoverTool(
        tooltips=[
            ('Date', '@date{%F}'),
            ('Open', '@open{0,0.00}'),
            ('High', '@high{0,0.00}'),
            ('Low', '@low{0,0.00}'),
            ('Close', '@close{0,0.00}')
        ],
        formatters={'@date': 'datetime'},
        mode='vline'
    )
    p.add_tools(hover)
    p.add_tools(CrosshairTool())
    
    # Configure legend
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    
    # Create statistics table with HTML/DIV approach for more reliable display
    from bokeh.models import Div
    
    stats_data = {
        'Metric': [
            'Return [%]', 'Buy & Hold Return [%]', 'Max. Drawdown [%]', 
            'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio',
            '# Trades', 'Win Rate [%]', 'Best Trade [%]', 'Worst Trade [%]'
        ],
        'Value': [
            f"{stats['Return [%]']:.2f}",
            f"{stats['Buy & Hold Return [%]']:.2f}",
            f"{stats['Max. Drawdown [%]']:.2f}",
            f"{stats['Sharpe Ratio']:.2f}",
            f"{stats['Sortino Ratio']:.2f}",
            f"{stats['Calmar Ratio']:.2f}",
            f"{stats['# Trades']}",
            f"{stats['Win Rate [%]']:.2f}",
            f"{stats['Best Trade [%]']:.2f}",
            f"{stats['Worst Trade [%]']:.2f}"
        ]
    }
    
    # Create HTML for statistics table
    stats_html = """
    <div style="padding: 15px; margin-top: 15px;">
        <h3 style="text-align: center; margin-bottom: 15px;">Backtest Statistics</h3>
        <table style="width: 100%; border-collapse: collapse; text-align: center;">
            <tr style="border-bottom: 1px solid #555;">
                <th style="padding: 8px;">Metric</th>
                <th style="padding: 8px;">Value</th>
                <th style="padding: 8px;">Metric</th>
                <th style="padding: 8px;">Value</th>
            </tr>
    """
    
    # Add rows for statistics
    metrics = stats_data['Metric']
    values = stats_data['Value']
    half = len(metrics) // 2
    
    for i in range(half):
        stats_html += f"""
        <tr style="border-bottom: 1px solid #444;">
            <td style="padding: 8px; font-weight: bold;">{metrics[i]}</td>
            <td style="padding: 8px;">{values[i]}</td>
            <td style="padding: 8px; font-weight: bold;">{metrics[i + half]}</td>
            <td style="padding: 8px;">{values[i + half]}</td>
        </tr>
        """
    
    stats_html += """
        </table>
    </div>
    """
    
    # Create Div component with the HTML
    stats_div = Div(text=stats_html, width=1000, height=300)
    
    # Layout with both components
    from bokeh.layouts import column
    layout = column(p, stats_div)
    
    # Convert to HTML
    html_content = file_html(layout, CDN, "Backtest Results")
    
    return html_content

# Initialize the app with dark theme
app = dash.Dash(
    __name__, 
    external_stylesheets=[
        dbc.themes.DARKLY,  # Using Bootstrap's dark theme
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.1/css/all.min.css"
    ],
    suppress_callback_exceptions=True
)

# Database functions
strategy_db = None

def get_strategy_db():
    """Connects to SQLite database and creates a strategy table if it doesn't exist."""
    global strategy_db
    if strategy_db:
        return strategy_db
    else:
        strategy_db = sqlite3.connect("strategy_db.sqlite", check_same_thread=False)
        cursor = strategy_db.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategy (
                id TEXT PRIMARY KEY,
                name TEXT,
                risk_return_preference REAL,
                indicators TEXT
            )
        """)
        strategy_db.commit()
        return strategy_db

def save_strategy_to_db(strategy_id, strategy_name, risk_return_preference, indicators_data):
    """Save strategy to database."""
    try:
        db = get_strategy_db()
        cursor = db.cursor()
        indicators_json = json.dumps(indicators_data)
        
        cursor.execute("SELECT id FROM strategy WHERE id = ?", (strategy_id,))
        if cursor.fetchone():
            cursor.execute(
                "UPDATE strategy SET name = ?, risk_return_preference = ?, indicators = ? WHERE id = ?",
                (strategy_name, risk_return_preference, indicators_json, strategy_id)
            )
        else:
            cursor.execute(
                "INSERT INTO strategy (id, name, risk_return_preference, indicators) VALUES (?, ?, ?, ?)",
                (strategy_id, strategy_name, risk_return_preference, indicators_json)
            )
        db.commit()
        return True
    except Exception as e:
        print(f"Error saving strategy: {e}")
        return False

def delete_strategy_from_db(strategy_id):
    """Delete strategy from database."""
    try:
        db = get_strategy_db()
        cursor = db.cursor()
        cursor.execute("DELETE FROM strategy WHERE id = ?", (strategy_id,))
        db.commit()
        return True
    except Exception as e:
        print(f"Error deleting strategy: {e}")
        return False

def load_strategies():
    """Load all strategies from database."""
    db = get_strategy_db()
    cursor = db.cursor()
    cursor.execute("SELECT id, name, risk_return_preference, indicators FROM strategy")
    strategies = []
    
    for row in cursor.fetchall():
        strategy_id, name, risk_return_preference, indicators_json = row
        indicators = json.loads(indicators_json) if indicators_json else []
        strategies.append({
            "id": strategy_id,
            "name": name,
            "risk_return_preference": risk_return_preference,
            "indicators": indicators
        })
    
    return strategies



def create_indicator_item(indicator_id):
    """Create an indicator selection box with a dynamic description panel."""
    return html.Div([
        # Indicator Selection Dropdown
        dbc.Select(
            id={'type': 'indicator-select', 'index': indicator_id},
            options=[
                {"label": "RSI", "value": "RSI"},
                {"label": "MACD", "value": "MACD"},
                {"label": "SMI", "value": "SMI"},
                {"label": "Stochastic Oscillator", "value": "STOCH"},
                {"label": "SMA Cross", "value": "SMA_CROSS"},
                {"label": "Bollinger Bands", "value": "BBANDS"}
            ],
            placeholder="Select an indicator",
            className="mb-2"
        ),

        # Description Panel
        html.Div(id={'type': 'indicator-description', 'index': indicator_id}, className="p-2 border rounded bg-dark text-light"),
        
        # Indicator Configuration Inputs
        html.Div(id={'type': 'indicator-config', 'index': indicator_id})
    ], className="mt-2 p-3 border rounded")

def create_strategy_card(strategy_id, strategy_name=None, risk_return_preference=1):
    """Create a strategy card with name editing, risk-return setting, and save functionality."""
    return dbc.Card([
        dbc.CardHeader(
            dbc.Row([
                dbc.Col(
                    dbc.Input(
                        id={'type': 'strategy-name-input', 'index': strategy_id},
                        type="text",
                        placeholder="Enter strategy name",
                        value=strategy_name,
                        className="mb-2"
                    ),
                    width=8
                ),
                dbc.Col(
                    dbc.Button(
                        [
                            html.I(className="fas fa-chevron-down", id={'type': 'strategy-icon', 'index': strategy_id})
                        ],
                        id={'type': 'strategy-collapse-button', 'index': strategy_id},
                        color="link",
                        className="text-end"
                    ),
                    width=4
                )
            ])
        ),
        dbc.Collapse(
            dbc.CardBody([
                html.Div([
                    # Risk-Return Preference Section
                   

                    # Row for Titles (Aligned with Inputs)
                    dbc.Row([
                        dbc.Col([
                            html.H5([
                                "Risk-Return Preference",
                                html.Span(
                                    "❓",
                                    id="risk-return-tooltip-target",
                                    style={"cursor": "pointer", "marginLeft": "8px", "fontSize": "16px", "color": "#17a2b8"}
                                ),
                            ]),
                            dbc.Tooltip(
                                "This metric represents your risk-reward ratio. A value of 2 means you risk 1% to potentially gain 2%. "
                                "Higher values indicate a more aggressive strategy, while lower values suggest a conservative approach.",
                                target="risk-return-tooltip-target",
                                placement="right",
                                className="tooltip-custom"
                            ),
                        ], width=6),
                    
                        dbc.Col([
                            html.H5([
                                "Position Size",
                                html.Span(
                                    "❓",
                                    id="position-size-tooltip-target",
                                    style={"cursor": "pointer", "marginLeft": "8px", "fontSize": "16px", "color": "#17a2b8"}
                                ),
                            ], style={"textAlign": "left"}),  # Ensures alignment
                            dbc.Tooltip(
                                "Defines the percentage of your total broker cash allocated per trade. "
                                "For example, a value of 0.3 means you risk 30% of your available capital on each buy. "
                                "This helps control risk and manage portfolio exposure.",
                                target="position-size-tooltip-target",
                                placement="right",
                                className="tooltip-custom"
                            ),
                        ], width=6)
                    ]),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Risk-Reward Ratio:", className="mb-2"),
                            dbc.Input(
                                id={'type': 'risk-return-input', 'index': strategy_id},
                                type="number",
                                min=0.1,
                                max=10,
                                step=0.1,
                                value=risk_return_preference,
                                className="mb-3"
                            ),
                        ], width=6),
                    
                        dbc.Col([
                            dbc.Label("Position Size (% of Broker Cash):", className="mb-2"),
                            dbc.Input(
                                id={'type': 'position-size-input', 'index': strategy_id},
                                type="number",
                                min=0.01,
                                max=1,
                                step=0.01,
                                value=0.3,  # Default to 30% of broker cash
                                className="mb-3"
                            ),
                        ], width=6),
                    ]),
                    
                    # Indicators Section
                    html.H5("Indicators", className="mt-3"),
                    html.Div(id={'type': 'indicators-container', 'index': strategy_id}, children=[]),
                    dbc.Row([
                        dbc.Col(
                            dbc.Button(
                                [html.I(className="fas fa-plus me-2"), "Add Indicator"],
                                id={'type': 'add-indicator-button', 'index': strategy_id},
                                color="primary",
                                size="sm",
                                className="mt-3"
                            ),
                            width="auto"
                        ),
                        dbc.Col(
                            dbc.Button(
                                [html.I(className="fas fa-save me-2"), "Save Strategy"],
                                id={'type': 'save-strategy-button', 'index': strategy_id},
                                color="success",
                                size="sm",
                                className="mt-3"
                            ),
                            width="auto",
                            className="ms-auto"
                        )
                    ])
                ])
            ]),
            id={'type': 'strategy-collapse', 'index': strategy_id},
            is_open=False,
        ),
    ], className="mb-3")
# Create the navbar with tabs
navbar = dbc.Tabs(
    [
        dbc.Tab(label="Strategy", tab_id="strategy"),
        dbc.Tab(label="Historical", tab_id="historical"),
        dbc.Tab(label="Live", tab_id="live"),
    ],
    id="tabs",
    active_tab="strategy",
)

# Page Layouts
strategy_layout = html.Div([
    html.H2("Strategy Builder", className="mt-3 mb-4"),
    dbc.Card([
        dbc.CardHeader("Strategies"),
        dbc.CardBody([
            html.Div(id="strategies-container", children=[]),
            dbc.Button(
                [html.I(className="fas fa-plus me-2"), "Add Strategy"],
                id="add-strategy-button",
                color="success",
                className="mt-3"
            ),
        ])
    ]),
    html.Div(id="strategies-table-container", className="mt-4")
])



from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA, GOOG
from bokeh.embed import file_html
from bokeh.resources import CDN


# Define SMA Crossover Strategy
class SmaCross(Strategy):
    n1 = 10
    n2 = 20

    def init(self):
        close = self.data.Close
        self.sma1 = self.I(SMA, close, self.n1)
        self.sma2 = self.I(SMA, close, self.n2)

    def next(self):
        if crossover(self.sma1, self.sma2):
            self.position.close()
            self.buy()
        elif crossover(self.sma2, self.sma1):
            self.position.close()
            self.sell()


from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, CrosshairTool, Legend
from bokeh.layouts import column
from bokeh.embed import file_html

def generate_custom_backtest_plot(cash):

    # Define SMA Crossover Strategy
    class SmaCross(Strategy):
        n1 = 10
        n2 = 20

        def init(self):
            close = self.data.Close
            self.sma1 = self.I(SMA, close, self.n1)
            self.sma2 = self.I(SMA, close, self.n2)

        def next(self):
            if crossover(self.sma1, self.sma2):
                self.position.close()
                self.buy()
            elif crossover(self.sma2, self.sma1):
                self.position.close()
                self.sell()

    # Run the backtest
    bt = Backtest(GOOG, SmaCross, cash=cash, commission=.002, exclusive_orders=True)
    stats = bt.run()
    
    # Get the data we need for plotting
    data = GOOG.copy()
    
    # Convert index to datetime if it's not already
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    
    # Add strategy indicators
    sma1 = SMA(data.Close, 10)
    sma2 = SMA(data.Close, 20)
    
    # Get trade data from stats
    trades = stats['_trades']
    
    # Create a Bokeh figure for the price chart
    p = figure(
        title="SMA Crossover Backtest",
        x_axis_type="datetime",
        width=1000,
        height=500,
        tools="pan,wheel_zoom,box_zoom,reset,save",
        toolbar_location="right"
    )
    
    # Create a ColumnDataSource for the OHLC data
    source = ColumnDataSource(data={
        'date': data.index,
        'open': data.Open,
        'high': data.High,
        'low': data.Low,
        'close': data.Close,
        'sma1': sma1,
        'sma2': sma2
    })
    
    # Plot price as a line
    price_line = p.line('date', 'close', source=source, color='#1F77B4', line_width=2, legend_label="Price")
    
    # Plot SMAs
    sma1_line = p.line('date', 'sma1', source=source, color='#FF7F0E', line_width=1.5, legend_label=f"SMA({SmaCross.n1})")
    sma2_line = p.line('date', 'sma2', source=source, color='#2CA02C', line_width=1.5, legend_label=f"SMA({SmaCross.n2})")
    
    # Add buy/sell markers if trades exist
    if not trades.empty:
        buy_signals = trades[trades.Size > 0]
        sell_signals = trades[trades.Size < 0]
        
        # Plot buy signals
        if not buy_signals.empty:
            p.circle(
                x=buy_signals.EntryTime,
                y=buy_signals.EntryPrice,
                size=10,
                color='green',
                alpha=0.7,
                legend_label="Buy"
            )
        
        # Plot sell signals
        if not sell_signals.empty:
            p.circle(
                x=sell_signals.EntryTime,
                y=sell_signals.EntryPrice,
                size=10,
                color='red',
                alpha=0.7,
                legend_label="Sell"
            )
    
    # Add hover tool
    hover = HoverTool(
        tooltips=[
            ('Date', '@date{%F}'),
            ('Open', '@open{0,0.00}'),
            ('High', '@high{0,0.00}'),
            ('Low', '@low{0,0.00}'),
            ('Close', '@close{0,0.00}'),
            (f'SMA({SmaCross.n1})', '@sma1{0,0.00}'),
            (f'SMA({SmaCross.n2})', '@sma2{0,0.00}')
        ],
        formatters={'@date': 'datetime'},
        mode='vline'
    )
    p.add_tools(hover)
    p.add_tools(CrosshairTool())
    
    # Configure legend
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    
    # Create a stats summary figure
    stats_data = {
        'Metric': [
            'Return [%]', 'Buy & Hold Return [%]', 'Max. Drawdown [%]', 
            'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio',
            '# Trades', 'Win Rate [%]', 'Best Trade [%]', 'Worst Trade [%]'
        ],
        'Value': [
            f"{stats['Return [%]']:.2f}",
            f"{stats['Buy & Hold Return [%]']:.2f}",
            f"{stats['Max. Drawdown [%]']:.2f}",
            f"{stats['Sharpe Ratio']:.2f}",
            f"{stats['Sortino Ratio']:.2f}",
            f"{stats['Calmar Ratio']:.2f}",
            f"{stats['# Trades']}",
            f"{stats['Win Rate [%]']:.2f}",
            f"{stats['Best Trade [%]']:.2f}",
            f"{stats['Worst Trade [%]']:.2f}"
        ]
    }
    
    # Create a stats table with improved layout
    from bokeh.models import ColumnDataSource, DataTable, TableColumn
    
    # Create a DataTable instead of a figure for statistics
    source = ColumnDataSource(data=dict(
        metric=stats_data['Metric'],
        value=stats_data['Value']
    ))
    
    columns = [
        TableColumn(field="metric", title="Metric"),
        TableColumn(field="value", title="Value")
    ]
    
    data_table = DataTable(
        source=source, 
        columns=columns,
        width=1000, 
        height=300,
        index_position=None
    )
    
    # Layout with table
    layout = column(p, data_table)
    
    # Convert to HTML
    html_content = file_html(layout, CDN, "Backtest Results")
    
    return html_content
# Historical Data Layout
historical_layout = html.Div([
    html.H2("Historical Data Backtesting", className="mt-3 mb-4"),

    dbc.Card([
        dbc.CardBody([
            dbc.Row([
                # Ticker input with label
                dbc.Col([
                    html.Label("Ticker",
                             style={
                                 'color': '#E0E0E0',  # Light gray to match theme
                                 'font-weight': '500',
                                 'margin-bottom': '4px',
                                 'display': 'block'
                             }),
                    dbc.Input(
                        id="ticker-input",
                        type="text",
                        placeholder="Enter Ticker (e.g., GOOG)",
                        style={
                            'height': '38px'
                        }
                    )
                ],
                width=2,
                className="mb-2"
                ),
               
                # Timeframe dropdown with label
                dbc.Col([
                    html.Label("Timeframe",
                             style={
                                 'color': '#E0E0E0',  # Light gray to match theme
                                 'font-weight': '500',
                                 'margin-bottom': '4px',
                                 'display': 'block'
                             }),
                    dcc.Dropdown(
                        id="timeframe-dropdown",
                        options=[
                            {"label": "1 Minute", "value": "1m"},
                            {"label": "5 Minutes", "value": "5m"},
                            {"label": "15 Minutes", "value": "15m"},
                            {"label": "1 Hour", "value": "1h"},
                            {"label": "1 Day", "value": "1d"},
                        ],
                        placeholder="Select Timeframe",
                        style={
                            'color': 'black',
                            'height': '38px'
                        }
                    )
                ],
                width=2,
                className="mb-2"
                ),
               
                # Start Date with label
                dbc.Col([
                    html.Label("Start Date",
                             style={
                                 'color': '#E0E0E0',  # Light gray to match theme
                                 'font-weight': '500',
                                 'margin-bottom': '4px',
                                 'display': 'block'
                             }),
                    dbc.Input(
                        id="start-date",
                        type="date",
                        style={
                            'height': '38px'
                        }
                    )
                ],
                width=2,
                className="mb-2"
                ),
               
                # End Date with label
                dbc.Col([
                    html.Label("End Date",
                             style={
                                 'color': '#E0E0E0',  # Light gray to match theme
                                 'font-weight': '500',
                                 'margin-bottom': '4px',
                                 'display': 'block'
                             }),
                    dbc.Input(
                        id="end-date",
                        type="date",
                        style={
                            'height': '38px'
                        }
                    )
                ],
                width=2,
                className="mb-2"
                ),
               
                # Strategy dropdown with label
                dbc.Col([
                    html.Label("Strategy",
                             style={
                                 'color': '#E0E0E0',  # Light gray to match theme
                                 'font-weight': '500',
                                 'margin-bottom': '4px',
                                 'display': 'block'
                             }),
                    dcc.Dropdown(
                        id="strategy-dropdown",
                        placeholder="Select Strategy",
                        style={
                            'color': 'black',
                            'height': '38px'
                        }
                    )
                ],
                width=2,
                className="mb-2"
                ),
               
                # Broker Cash input with label
                dbc.Col([
                    html.Label("Broker Cash",
                             style={
                                 'color': '#E0E0E0',  # Light gray to match theme
                                 'font-weight': '500',
                                 'margin-bottom': '4px',
                                 'display': 'block'
                             }),
                    dbc.Input(
                        id="cash-input",
                        type="number",
                        placeholder="Enter amount",
                        style={
                            'height': '38px'
                        }
                    )
                ],
                width=2,
                className="mb-2"
                ),
            ], className="g-2"),
           
            dbc.Row([
                dbc.Col(
                    dbc.Button(
                        "Run Backtest",
                        id="run-backtest-btn",
                        color="primary",
                        className="mt-2"
                    ),
                    width="auto"
                ),
            ], className="mt-2"),
        ])
    ], className="mb-3"),

    # BACKTEST PLOT
    dbc.Row([
        dbc.Col(html.Iframe(id="backtest-bokeh-frame", style={"width": "100%", "height": "600px", "border": "none"}), width=12)
    ])
])



# Callback for indicator description links
@callback(
    Output({'type': 'indicator-description', 'index': MATCH}, 'children'),
    Input({'type': 'indicator-select', 'index': MATCH}, 'value'),
    prevent_initial_call=True
)
def update_indicator_description(indicator_type):
    """Provides an explanation for each selected indicator with an Investopedia link."""
    descriptions = {
        "RSI": html.Span([
            "Relative Strength Index (RSI) measures momentum. Values above 70 indicate overbought conditions, while values below 30 indicate oversold levels. ",
            html.A("Learn more", href="https://www.investopedia.com/terms/r/rsi.asp", target="_blank", style={"color": "#17a2b8", "textDecoration": "none"}),
            " on Investopedia."
        ]),
        "MACD": html.Span([
            "Moving Average Convergence Divergence (MACD) is a trend-following momentum indicator that shows the relationship between two moving averages. ",
            html.A("Learn more", href="https://www.investopedia.com/terms/m/macd.asp", target="_blank", style={"color": "#17a2b8", "textDecoration": "none"}),
            " on Investopedia."
        ]),
        "SMI": html.Span([
            "Stochastic Momentum Index (SMI) is a refined version of the Stochastic Oscillator, focusing on market momentum shifts. ",
            html.A("Learn more", href="https://www.investopedia.com/terms/s/stochastic-momentum-index-smi.asp", target="_blank", style={"color": "#17a2b8", "textDecoration": "none"}),
            " on Investopedia."
        ]),
        "STOCH": html.Span([
            "Stochastic Oscillator compares a security’s closing price to its price range over a given period, helping identify overbought and oversold conditions. ",
            html.A("Learn more", href="https://www.investopedia.com/terms/s/stochasticoscillator.asp", target="_blank", style={"color": "#17a2b8", "textDecoration": "none"}),
            " on Investopedia."
        ]),
        "SMA_CROSS": html.Span([
            "Simple Moving Average (SMA) Cross identifies trend changes when a short-term SMA crosses a long-term SMA. ",
            html.A("Learn more", href="https://www.investopedia.com/terms/s/sma.asp", target="_blank", style={"color": "#17a2b8", "textDecoration": "none"}),
            " on Investopedia."
        ]),
        "BBANDS": html.Span([
            "Bollinger Bands consist of a moving average with upper and lower bands based on standard deviation, indicating volatility. ",
            html.A("Learn more", href="https://www.investopedia.com/terms/b/bollingerbands.asp", target="_blank", style={"color": "#17a2b8", "textDecoration": "none"}),
            " on Investopedia."
        ]),
    }
    return descriptions.get(indicator_type, "Select an indicator to see its definition.")



#Dropdown to select the strategy in the historical tab
@callback(
    Output("strategy-dropdown", "options"),
    Input("table-trigger", "data")
)
def update_historical_strategy_dropdown(trigger):
    strategies = load_strategies()
    return [{"label": s["name"] or "Unnamed Strategy", "value": s["id"]} for s in strategies] if strategies else []


@callback(
    Output("backtest-bokeh-frame", "srcDoc"),
    Input("run-backtest-btn", "n_clicks"),
    [State("ticker-input", "value"),
     State("timeframe-dropdown", "value"),
     State("start-date", "value"),
     State("end-date", "value"),
     State("strategy-dropdown", "value"),
     State("cash-input", "value")],
    prevent_initial_call=True
)
def run_backtest(n_clicks, ticker, timeframe, start_date, end_date, strategy_id, cash):
    if not all([ticker, timeframe, start_date, end_date, strategy_id, cash]):
        # Create error message if inputs are missing
        from bokeh.plotting import figure
        from bokeh.embed import file_html
        from bokeh.resources import CDN
        
        error_fig = figure(title="Missing Input Parameters", width=1000, height=300)
        error_fig.text(
            x=0.5, y=0.5, 
            text=["Please provide all required inputs: ticker, timeframe, dates, strategy and cash amount."],
            text_align="center", text_baseline="middle", text_font_size="14px"
        )
        return file_html(error_fig, CDN)
    
    try:
        # Convert cash to float
        cash = float(cash)
        
        # Fetch data from Alpaca API
        api = tradeapi.REST(default_key, default_secret_key, BASE_URL, api_version='v2')
        
        # Convert timeframe to Alpaca format
        timeframe_map = {
            "1m": "1Min",
            "5m": "5Min", 
            "15m": "15Min",
            "1h": "1Hour",
            "1d": "1Day"
        }
        alpaca_timeframe = timeframe_map.get(timeframe, "1Day")
        
        # Fetch historical data
        logger.info(f"Fetching data for {ticker} from {start_date} to {end_date}")
        bars = api.get_bars(
            symbol=ticker,
            timeframe=alpaca_timeframe,
            start=start_date,
            end=end_date
        ).df
        
        if bars.empty:
            raise ValueError(f"No data available for {ticker} in selected date range")
        
        # Prepare data for backtesting
        data = pd.DataFrame({
            'Open': bars['open'],
            'High': bars['high'],
            'Low': bars['low'],
            'Close': bars['close'],
            'Volume': bars['volume']
        })
        
        # Get strategy configuration from database
        db = get_strategy_db()
        cursor = db.cursor()
        cursor.execute("SELECT id, name, risk_return_preference, indicators FROM strategy WHERE id = ?", (strategy_id,))
        strategy_row = cursor.fetchone()
        
        if not strategy_row:
            raise ValueError(f"Strategy with ID {strategy_id} not found")
        
        strategy_config = {
            "id": strategy_row[0],
            "name": strategy_row[1],
            "risk_return_preference": float(strategy_row[2]),
            "indicators": json.loads(strategy_row[3])
        }
        
        # Create custom strategy class
        CustomStrategy = create_custom_strategy_class(strategy_config)
        
        # Run backtest
        bt = Backtest(data, CustomStrategy, cash=cash, commission=0.002, exclusive_orders=True)
        stats = bt.run()
        
        # Generate visualization
        return generate_backtest_plot(data, stats, strategy_config)
        
    except Exception as e:
        logger.error(f"Backtest error: {str(e)}", exc_info=True)
        
        # Create simple error message figure
        from bokeh.plotting import figure
        from bokeh.embed import file_html
        from bokeh.resources import CDN
        
        error_fig = figure(title=f"Error Running Backtest", width=1000, height=300)
        error_fig.text(
            x=0.5, y=0.5, 
            text=[f"An error occurred: {str(e)}"],
            text_align="center", text_baseline="middle", text_font_size="14px"
        )
        return file_html(error_fig, CDN)

# Live Trading Layout
live_layout = html.Div([
    html.H2("Live Trading", className="mt-3 mb-4"),
    
    #Refresh the table
    # Add this to your live_layout at the beginning
    dcc.Interval(
        id="interval-component",
        interval=30000,  # 30 seconds in milliseconds
        n_intervals=0
    ),
    # Portfolio Performance Section
    dbc.Card([
        dbc.CardHeader(html.H5("Portfolio Performance", className="mb-0")),
        dbc.CardBody([
            dbc.Row([
                # Left column for key metrics
                dbc.Col([
                    html.Div([
                        html.H6("Account Summary", className="mb-3"),
                        dbc.Row([
                            dbc.Col(html.Div("Portfolio Value:", className="fw-bold"), width=6),
                            dbc.Col(html.Div(id="portfolio-value", children="$0.00"), width=6),
                        ], className="mb-2"),
                        dbc.Row([
                            dbc.Col(html.Div("Cash Balance:", className="fw-bold"), width=6),
                            dbc.Col(html.Div(id="cash-balance", children="$0.00"), width=6),
                        ], className="mb-2"),
                        dbc.Row([
                            dbc.Col(html.Div("Today's P/L:", className="fw-bold"), width=6),
                            dbc.Col(html.Div(id="daily-pl", children="$0.00"), width=6),
                        ], className="mb-2"),
                        dbc.Row([
                            dbc.Col(html.Div("Total P/L:", className="fw-bold"), width=6),
                            dbc.Col(html.Div(id="total-pl", children="$0.00"), width=6),
                        ], className="mb-2"),
                        html.Hr(),
                        dbc.Row([
                            dbc.Col([
                                dbc.Button(
                                    [html.I(className="fas fa-sync-alt me-2"), "Refresh Data"],
                                    id="refresh-portfolio-btn",
                                    color="secondary",
                                    size="sm",
                                    className="w-100"
                                )
                            ], width=12)
                        ])
                    ], className="p-2")
                ], md=3),
                
                # Right column for chart
                dbc.Col([
                    html.Div(
                        dcc.Graph(
                            id="portfolio-chart",
                            figure={
                                'data': [{
                                    'x': [],
                                    'y': [],
                                    'type': 'line',
                                    'name': 'Portfolio Value'
                                }],
                                'layout': {
                                    'title': 'Portfolio Performance Over Time',
                                    'paper_bgcolor': 'rgba(0,0,0,0)',
                                    'plot_bgcolor': 'rgba(0,0,0,0)',
                                    'font': {'color': 'white'},
                                    'xaxis': {'gridcolor': '#444444'},
                                    'yaxis': {'gridcolor': '#444444'}
                                }
                            },
                            config={'displayModeBar': False}
                        ),
                        className="chart-container"
                    )
                ], md=9)
            ])
        ])
    ], className="mb-4"),
        
    # Strategy Execution Section
    dbc.Card([
        dbc.CardHeader(html.H5("Strategy Execution", className="mb-0")),
        dbc.CardBody([
            dbc.Row([
                # Strategy selection dropdown
                dbc.Col([
                    dbc.Label("Select Strategy:"),
                    dcc.Dropdown(
                        id="live-strategy-dropdown",
                        options=[],  # Will be populated from the database
                        placeholder="Choose a strategy to execute",
                        style={'color': 'black'}
                    )
                ], md=4),
                
                # Alpaca API configuration
                dbc.Col([
                    dbc.Label("Alpaca API Settings:"),
                    dbc.Row([
                        dbc.Col(
                            dbc.Input(id="alpaca-api-key", type="password", placeholder="API Key", className="mb-2"),
                            width=6
                        ),
                        dbc.Col(
                            dbc.Input(id="alpaca-api-secret", type="password", placeholder="API Secret", className="mb-2"),
                            width=6
                        )
                    ])
                ], md=4),

                # Symbol Input
                dbc.Col([
                    dbc.Label("Symbol:"),
                    dbc.Input(id="symbol-input", type="text", placeholder="Enter Symbol (e.g., AAPL)", className="mb-2")
                ], md=4)
            ], className="mb-3"),
            
            # Action buttons
            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        [html.I(className="fas fa-play me-2"), "Start Trading"],
                        id="start-trading-btn",
                        color="success",
                        className="me-2"
                    ),
                    dbc.Button(
                        [html.I(className="fas fa-stop me-2"), "Stop Trading"],
                        id="stop-trading-btn",
                        color="danger",
                        disabled=True
                    )
                ], width="auto"),
                dbc.Col([
                    # Trading Status Indicator
                    html.Div([
                        html.Span("Status: ", className="me-2"),
                        html.Span("Inactive", id="trading-status", className="text-warning")
                    ], className="d-flex align-items-center h-100")
                ], width="auto"),
                dbc.Col([
                    # Last Update Time
                    html.Div([
                        html.Span("Last Update: ", className="me-2"),
                        html.Span("Never", id="last-update-time")
                    ], className="d-flex align-items-center h-100")
                ], width="auto")
            ], className="mb-3"),
        ])
    ], className="mb-4"),

    dbc.Card([
        dbc.CardHeader(html.H5("Trading Log", className="mb-0")),
        dbc.CardBody([
            html.Div(id="trading-logs", style={"height": "200px", "overflow": "auto"}),
            dbc.ButtonGroup([
                dbc.Button(
                    [html.I(className="fas fa-sync-alt me-2"), "Refresh Logs"],
                    id="refresh-logs-btn",
                    color="secondary",
                    size="sm",
                    className="mt-2"
                ),
                dbc.Button(
                    [html.I(className="fas fa-trash me-2"), "Clear Logs"],
                    id="clear-logs-btn", 
                    color="danger",
                    size="sm",
                    className="mt-2"
                ),
            ]),
        ])
    ], className="mb-4"),

    # Orders Table Section
    dbc.Card([
        dbc.CardHeader(
            dbc.Row([
                dbc.Col(html.H5("Recent Orders", className="mb-0"), width="auto"),
                dbc.Col(
                    dbc.Button(
                        [html.I(className="fas fa-sync-alt me-2"), "Refresh"],
                        id="refresh-orders-btn",
                        color="link",
                        size="sm",
                        className="float-end text-decoration-none"
                    ),
                    width="auto",
                    className="ms-auto"
                )
            ])
        ),
        dbc.CardBody([
            html.Div(
                id="orders-table-container",
                children=[
                    dbc.Table(
                        [
                       html.Thead(
                            html.Tr([
                                html.Th("Symbol"),
                                html.Th("Side"),
                                html.Th("Price"),
                                html.Th("Filled At"),
                            ])
                        ),
                            html.Tbody(id="orders-table-body", children=[
                                html.Tr([
                                    html.Td(className="text-center"),
                                    html.Td("No orders found. Start trading to see orders here.")
                                ])
                            ])
                        ],
                        bordered=True,
                        hover=True,
                        responsive=True,
                        className="orders-table"
                    )
                ]
            )
        ])
    ])
])




# Main layout
app.layout = html.Div([
    dbc.Container([
        html.H1("QuantForge", className="mt-3 mb-3"),
        navbar,
        html.Div(id="page-content", className="mt-3"),
        dcc.Store(id='table-trigger', data=0)
    ])
])


#Logger callback
@callback(
    Output("trading-logs", "children", allow_duplicate=True),
    Input("refresh-logs-btn", "n_clicks"),
    Input("start-trading-btn", "n_clicks"),
    prevent_initial_call=True
)
def update_trading_logs(refresh_clicks, start_clicks):
    try:
        with open("trading.log", "r") as f:
            logs = f.readlines()[-20:]  # Get last 20 lines
            return [html.P(log, style={"margin": "0", "padding": "0"}) for log in logs]
    except Exception as e:
        return [html.P(f"Error loading logs: {str(e)}")]
#Clearing logs
@callback(
    Output("trading-logs", "children"),
    Input("clear-logs-btn", "n_clicks"),
    prevent_initial_call=True
)
def clear_trading_logs(n_clicks):
    if n_clicks:
        try:
            # Open in write mode to truncate the file
            with open("trading.log", "w") as f:
                f.write("Log cleared at " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
                
            return [html.P("Log cleared", style={"margin": "0", "padding": "0"})]
        except Exception as e:
            return [html.P(f"Error clearing logs: {str(e)}", style={"margin": "0", "padding": "0"})]
    return dash.no_update

# Callback to update order tables
@callback(
    Output("orders-table-body", "children"),
    [Input("refresh-orders-btn", "n_clicks"),
     Input("start-trading-btn", "n_clicks"),
     Input("interval-component", "n_intervals")],
    [State("alpaca-api-key", "value"),
     State("alpaca-api-secret", "value")],
    prevent_initial_call=True
)
def update_orders_table(refresh_clicks, start_clicks, interval, api_key, api_secret):
    # Use provided keys or fall back to defaults
    api_key = api_key if api_key else default_key
    api_secret = api_secret if api_secret else default_secret_key
    
    if not api_key or not api_secret:
        return [html.Tr([html.Td(colSpan=4, className="text-center"), 
                       html.Td("API credentials required to fetch orders.")])]  
    try:
        # Initialize API with basic configuration
        api = tradeapi.REST(api_key, api_secret, BASE_URL, api_version='v2')
        
        # Use raw HTTP request to bypass date parsing issues
        import requests
        import json
        
        # Construct headers manually
        headers = {
            'APCA-API-KEY-ID': api_key,
            'APCA-API-SECRET-KEY': api_secret
        }
        
        # Make direct API request instead
        url = f"{BASE_URL}/v2/orders?status=all&limit=100"
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            return [html.Tr([html.Td(colSpan=4, className="text-center"), 
                          html.Td(f"Error {response.status_code}: {response.text}")])]
        
        all_orders = json.loads(response.text)
        
        # Filter for only filled orders
        filled_orders = [order for order in all_orders if order.get('status') == 'filled']
        
        if not filled_orders:
            return [html.Tr([html.Td(colSpan=4, className="text-center"), 
                           html.Td("No filled orders found.")])]
        
        # Format orders for display
        order_rows = []
        for order in filled_orders:
            # Apply color based on side
            side_color = "text-success" if order.get('side') == 'buy' else "text-danger"
            
            # Format price with dollar sign
            filled_price = order.get('filled_avg_price')
            price = f"${float(filled_price):.2f}" if filled_price else "-"
            
            # Format the filled time to be more readable
            filled_at = order.get('filled_at')
            filled_time = filled_at.replace('T', ' ').split('.')[0] if filled_at else "-"
            
            order_rows.append(html.Tr([
                html.Td(order.get('symbol')),
                html.Td(order.get('side', '').upper(), className=side_color),
                html.Td(price),
                html.Td(filled_time)
            ]))
        
        return order_rows
        
    except Exception as e:
        logger.error(f"Error fetching orders: {str(e)}")
        return [html.Tr([html.Td(colSpan=4, className="text-center"), 
                       html.Td(f"Error fetching orders: {str(e)}")])]

# Callback to update strategy dropdown
@callback(
    Output("live-strategy-dropdown", "options"),
    Input("table-trigger", "data")
)
def update_live_strategy_dropdown(trigger):
    strategies = load_strategies()
    return [{"label": s["name"], "value": s["id"]} for s in strategies] if strategies else []

# Callback for Start/Stop trading buttons state
# Historical Data Layout
historical_layout = html.Div([
    html.H2("Historical Data Backtesting", className="mt-3 mb-4"),

    dbc.Card([
        dbc.CardBody([
            dbc.Row([
                # Ticker input with label
                dbc.Col([
                    html.Label("Ticker",
                             style={
                                 'color': '#E0E0E0',  # Light gray to match theme
                                 'font-weight': '500',
                                 'margin-bottom': '4px',
                                 'display': 'block'
                             }),
                    dbc.Input(
                        id="ticker-input",
                        type="text",
                        placeholder="Enter Ticker (e.g., GOOG)",
                        style={
                            'height': '38px'
                        }
                    )
                ],
                width=2,
                className="mb-2"
                ),
               
                # Timeframe dropdown with label
                dbc.Col([
                    html.Label("Timeframe",
                             style={
                                 'color': '#E0E0E0',  # Light gray to match theme
                                 'font-weight': '500',
                                 'margin-bottom': '4px',
                                 'display': 'block'
                             }),
                    dcc.Dropdown(
                        id="timeframe-dropdown",
                        options=[
                            {"label": "1 Minute", "value": "1m"},
                            {"label": "5 Minutes", "value": "5m"},
                            {"label": "15 Minutes", "value": "15m"},
                            {"label": "1 Hour", "value": "1h"},
                            {"label": "1 Day", "value": "1d"},
                        ],
                        placeholder="Select Timeframe",
                        style={
                            'color': 'black',
                            'height': '38px'
                        }
                    )
                ],
                width=2,
                className="mb-2"
                ),
               
                # Start Date with label
                dbc.Col([
                    html.Label("Start Date",
                             style={
                                 'color': '#E0E0E0',  # Light gray to match theme
                                 'font-weight': '500',
                                 'margin-bottom': '4px',
                                 'display': 'block'
                             }),
                    dbc.Input(
                        id="start-date",
                        type="date",
                        style={
                            'height': '38px'
                        }
                    )
                ],
                width=2,
                className="mb-2"
                ),
               
                # End Date with label
                dbc.Col([
                    html.Label("End Date",
                             style={
                                 'color': '#E0E0E0',  # Light gray to match theme
                                 'font-weight': '500',
                                 'margin-bottom': '4px',
                                 'display': 'block'
                             }),
                    dbc.Input(
                        id="end-date",
                        type="date",
                        style={
                            'height': '38px'
                        }
                    )
                ],
                width=2,
                className="mb-2"
                ),
               
                # Strategy dropdown with label
                dbc.Col([
                    html.Label("Strategy",
                             style={
                                 'color': '#E0E0E0',  # Light gray to match theme
                                 'font-weight': '500',
                                 'margin-bottom': '4px',
                                 'display': 'block'
                             }),
                    dcc.Dropdown(
                        id="strategy-dropdown",
                        placeholder="Select Strategy",
                        style={
                            'color': 'black',
                            'height': '38px'
                        }
                    )
                ],
                width=2,
                className="mb-2"
                ),
               
                # Broker Cash input with label
                dbc.Col([
                    html.Label("Broker Cash",
                             style={
                                 'color': '#E0E0E0',  # Light gray to match theme
                                 'font-weight': '500',
                                 'margin-bottom': '4px',
                                 'display': 'block'
                             }),
                    dbc.Input(
                        id="cash-input",
                        type="number",
                        placeholder="Enter amount",
                        style={
                            'height': '38px'
                        }
                    )
                ],
                width=2,
                className="mb-2"
                ),
            ], className="g-2"),
           
            dbc.Row([
                dbc.Col(
                    dbc.Button(
                        "Run Backtest",
                        id="run-backtest-btn",
                        color="primary",
                        className="mt-2"
                    ),
                    width="auto"
                ),
            ], className="mt-2"),
        ])
    ], className="mb-3"),

    # BACKTEST PLOT
    dbc.Row([
        dbc.Col(html.Iframe(id="backtest-bokeh-frame", style={"width": "100%", "height": "600px", "border": "none"}), width=12)
    ])
])

# Update chart data with Alpaca 
@callback(
    [Output("portfolio-value", "children"),
     Output("cash-balance", "children"),
     Output("daily-pl", "children"),
     Output("total-pl", "children"),
     Output("last-update-time", "children"),
     Output("portfolio-chart", "figure")],
    [Input("refresh-portfolio-btn", "n_clicks"),
     Input("interval-component", "n_intervals")],
    [State("alpaca-api-key", "value"),
     State("alpaca-api-secret", "value")],
    prevent_initial_call=True
)
def update_portfolio_data(refresh_clicks, interval, api_key, api_secret):
    # Use default keys if none provided
    api_key = api_key if api_key else default_key
    api_secret = api_secret if api_key else default_secret_key
    
    if not api_key or not api_secret:
        empty_figure = {
            'data': [{
                'x': [],
                'y': [],
                'type': 'line',
                'name': 'Portfolio Value'
            }],
            'layout': {
                'title': 'Portfolio Performance Over Time',
                'paper_bgcolor': 'rgba(0,0,0,0)',
                'plot_bgcolor': 'rgba(0,0,0,0)',
                'font': {'color': 'white'},
                'xaxis': {'gridcolor': '#444444'},
                'yaxis': {'gridcolor': '#444444'}
            }
        }
        return "$0.00", "$0.00", "$0.00", "$0.00", "API credentials required", empty_figure
    
    try:
        # Initialize API
        api = tradeapi.REST(api_key, api_secret, BASE_URL, api_version='v2')
        
        # Get account information
        account = api.get_account()
        
        # Format portfolio metrics
        portfolio_value = f"${float(account.portfolio_value):,.2f}"
        cash_balance = f"${float(account.cash):,.2f}"
        
        # Calculate P/L
        daily_pl_value = float(account.equity) - float(account.last_equity)
        daily_pl_color = "text-success" if daily_pl_value >= 0 else "text-danger"
        daily_pl = html.Span(
            f"${abs(daily_pl_value):,.2f} ({abs(daily_pl_value)/float(account.last_equity)*100:.2f}%)",
            className=daily_pl_color
        )
        
        # Calculate total P/L based on initial investment (estimate)
        # This is simplified - you might need to calculate this differently
        total_pl_value = float(account.equity) - float(account.initial_margin)
        total_pl_color = "text-success" if total_pl_value >= 0 else "text-danger"
        total_pl = html.Span(
            f"${abs(total_pl_value):,.2f}",
            className=total_pl_color
        )
        
        # Get portfolio history for the chart
        end = pd.Timestamp.now(tz='America/New_York')
        start = end - pd.Timedelta(days=30)  # Last 30 days
        
        portfolio_history = api.get_portfolio_history(
            period='1M',  # 1 month
            timeframe='1D',  # Daily data points
            extended_hours=True
        )
        
        # Create dataframe from portfolio history
        timestamps = pd.to_datetime(portfolio_history.timestamp, unit='s')
        equity_values = portfolio_history.equity
        
        # Create the figure
        figure = {
            'data': [{
                'x': timestamps,
                'y': equity_values,
                'type': 'line',
                'name': 'Portfolio Value'
            }],
            'layout': {
                'title': 'Portfolio Performance (Last 30 Days)',
                'paper_bgcolor': 'rgba(0,0,0,0)',
                'plot_bgcolor': 'rgba(0,0,0,0)',
                'font': {'color': 'white'},
                'xaxis': {'gridcolor': '#444444'},
                'yaxis': {'gridcolor': '#444444', 'tickprefix': '$'}
            }
        }
        
        # Format current time
        current_time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return portfolio_value, cash_balance, daily_pl, total_pl, current_time, figure
        
    except Exception as e:
        logger.error(f"Error updating portfolio data: {str(e)}")
        empty_figure = {
            'data': [{
                'x': [],
                'y': [],
                'type': 'line',
                'name': 'Portfolio Value'
            }],
            'layout': {
                'title': f'Error: {str(e)}',
                'paper_bgcolor': 'rgba(0,0,0,0)',
                'plot_bgcolor': 'rgba(0,0,0,0)',
                'font': {'color': 'white'},
                'xaxis': {'gridcolor': '#444444'},
                'yaxis': {'gridcolor': '#444444'}
            }
        }
        return "$0.00", "$0.00", "$0.00", "$0.00", f"Error: {str(e)}", empty_figure

@callback(
    Output("page-content", "children"),
    Input("tabs", "active_tab"),
)
def switch_tab(tab):
    if tab == "strategy":
        return strategy_layout
    elif tab == "historical":
        return historical_layout
    elif tab == "live":
        return live_layout
    return html.P("Unknown tab selected")

@callback(
    Output("strategies-container", "children"),
    Input("add-strategy-button", "n_clicks"),
    State("strategies-container", "children"),
    prevent_initial_call=True
)
def add_strategy(n_clicks, existing_strategies):
    if n_clicks:
        strategy_id = str(uuid.uuid4())[:8]
        new_strategy = create_strategy_card(strategy_id)
        return existing_strategies + [new_strategy] if existing_strategies else [new_strategy]
    return existing_strategies or []

@callback(
    [Output({'type': 'strategy-collapse', 'index': MATCH}, 'is_open'),
     Output({'type': 'strategy-icon', 'index': MATCH}, 'className')],
    Input({'type': 'strategy-collapse-button', 'index': MATCH}, 'n_clicks'),
    State({'type': 'strategy-collapse', 'index': MATCH}, 'is_open'),
    prevent_initial_call=True
)
def toggle_strategy_collapse(n_clicks, is_open):
    new_open_state = not is_open
    icon_class = "fas fa-chevron-up" if new_open_state else "fas fa-chevron-down"
    return new_open_state, icon_class

@callback(
    Output({'type': 'indicators-container', 'index': MATCH}, 'children'),
    Input({'type': 'add-indicator-button', 'index': MATCH}, 'n_clicks'),
    State({'type': 'indicators-container', 'index': MATCH}, 'children'),
    prevent_initial_call=True
)
def add_indicator(n_clicks, existing_indicators):
    if n_clicks:
        indicator_id = str(uuid.uuid4())[:8]
        new_indicator = create_indicator_item(indicator_id)
        return existing_indicators + [new_indicator] if existing_indicators else [new_indicator]
    return existing_indicators or []

@callback(
    Output({'type': 'indicator-config', 'index': MATCH}, 'children'),
    Input({'type': 'indicator-select', 'index': MATCH}, 'value'),
    prevent_initial_call=True
)
def update_indicator_config(indicator_type):
    if not indicator_type:
        return html.Div("Select an indicator to configure its parameters")
        
    if indicator_type == "RSI":
        return html.Div([
            dbc.Label("Period"),
            dbc.Input(type="number", value=14, min=1, max=100, id={'type': 'RSI-period'}),
            dbc.Label("Overbought Level"),
            dbc.Input(type="number", value=70, min=1, max=100, id={'type': 'RSI-overbought'}),
            dbc.Label("Oversold Level"),
            dbc.Input(type="number", value=30, min=1, max=100, id={'type': 'RSI-oversold'}),
        ])
    elif indicator_type == "MACD":
        return html.Div([
            dbc.Label("Fast Period"),
            dbc.Input(type="number", value=12, min=1, max=100, id={'type': 'MACD-fast'}),
            dbc.Label("Slow Period"),
            dbc.Input(type="number", value=26, min=1, max=100, id={'type': 'MACD-slow'}),
            dbc.Label("Signal Period"),
            dbc.Input(type="number", value=9, min=1, max=100, id={'type': 'MACD-signal'}),
        ])
    elif indicator_type == "STOCH":
        return html.Div([
            dbc.Label("K Period"),
            dbc.Input(type="number", value=14, min=1, max=100, id={'type': 'STOCH-k'}),
            dbc.Label("D Period"),
            dbc.Input(type="number", value=3, min=1, max=100, id={'type': 'STOCH-d'}),
            dbc.Label("Slow Period"),
            dbc.Input(type="number", value=3, min=1, max=100, id={'type': 'STOCH-slow'}),
        ])
    elif indicator_type == "SMA_CROSS":
        return html.Div([
            dbc.Label("Fast Period"),
            dbc.Input(type="number", value=10, min=1, max=100, id={'type': 'SMA-fast'}),
            dbc.Label("Slow Period"),
            dbc.Input(type="number", value=20, min=1, max=100, id={'type': 'SMA-slow'}),
        ])
    elif indicator_type == "BBANDS":
        return html.Div([
            dbc.Label("Period"),
            dbc.Input(type="number", value=20, min=1, max=100, id={'type': 'BBANDS-period'}),
            dbc.Label("Standard Deviation"),
            dbc.Input(type="number", value=2, min=0.1, max=10, step=0.1, id={'type': 'BBANDS-std-dev'}),
        ])
    elif indicator_type == "SMI":
        return html.Div([
            dbc.Label("Period"),
            dbc.Input(type="number", value=14, min=1, max=100, id={'type': 'SMI-period'}),
            dbc.Label("Signal Period"),
            dbc.Input(type="number", value=9, min=1, max=100, id={'type': 'SMI-signal'}),
        ])
    return html.Div("Select an indicator to configure its parameters")
    # Other indicator types follow the same pattern...
@callback(
    Output({'type': 'save-strategy-button', 'index': MATCH}, "children"),
    Input({'type': 'save-strategy-button', 'index': MATCH}, "n_clicks"),
    [State({'type': 'strategy-name-input', 'index': MATCH}, "value"),
     State({'type': 'risk-return-input', 'index': MATCH}, "value"),
     State({'type': 'indicator-select', 'index': ALL}, "value"),
     State({'type': 'indicator-select', 'index': ALL}, "id"),
     State({'type': 'indicator-config', 'index': ALL}, "children")],
    prevent_initial_call=True
)
def handle_save_button(n_clicks, strategy_name, risk_return_preference, indicator_types, indicator_ids, indicator_configs):
 # Revised helper function that recursively extracts input values
    def extract_input_values(component):
        # Normalize: if component is a dict, get its children from its "props", else if already a list, use it.
        if isinstance(component, dict):
            children = component.get("props", {}).get("children", [])
            if not isinstance(children, list):
                children = [children]
        elif isinstance(component, list):
            children = component
        else:
            children = []

        vals = []
        for child in children:
            if isinstance(child, dict):
                # If this is an Input, grab its value
                if child.get("type") == "Input":
                    val = child.get("props", {}).get("value")
                    if val is not None:
                        vals.append(val)
                else:
                    # Otherwise, check whether it has nested children. (Labels and other components will be skipped.)
                    vals.extend(extract_input_values(child))
            # In case a child is itself a list, drill down further.
            elif isinstance(child, list):
                vals.extend(extract_input_values(child))
        return vals

    # Only proceed if there have been clicks.
    if not n_clicks:
        return dash.no_update

    # Determine the strategy id from the save button's id (using the MATCH pattern)
    ctx = dash.callback_context
    triggered_prop = ctx.triggered[0]['prop_id']
    try:
        strategy_id = json.loads(triggered_prop.split('.')[0])['index']
    except Exception as e:
        print("Error parsing strategy id:", e)
        return [html.I(className="fas fa-exclamation-triangle me-2"), "Error"]

    indicators_data = []
    # Here we assume the order of indicator_configs matches the order of indicator_types.
    for i, config in enumerate(indicator_configs):
        try:
            indicator_type = indicator_types[i]
        except IndexError:
            continue

        if not indicator_type:
            continue

        # Use our helper to get all input values (even if nested inside Divs or other containers)
        input_values = extract_input_values(config)

        # Based on what indicator type is selected, create the settings dictionary if sufficient values exist.
        settings = {}
        if indicator_type == "RSI" and len(input_values) >= 3:
            settings = {
                "period": input_values[0],
                "overbought": input_values[1],
                "oversold": input_values[2],
            }
        elif indicator_type == "MACD" and len(input_values) >= 3:
            settings = {
                "fast_period": input_values[0],
                "slow_period": input_values[1],
                "signal_period": input_values[2],
            }
        elif indicator_type == "STOCH" and len(input_values) >= 3:
            settings = {
                "k_period": input_values[0],
                "d_period": input_values[1],
                "slow_period": input_values[2],
            }
        elif indicator_type == "SMA_CROSS" and len(input_values) >= 2:
            settings = {
                "fast_period": input_values[0],
                "slow_period": input_values[1],
            }
        elif indicator_type == "BBANDS" and len(input_values) >= 2:
            settings = {
                "period": input_values[0],
                "std_dev": input_values[1],
            }
        elif indicator_type == "SMI" and len(input_values) >= 2:
            settings = {
                "period": input_values[0],
                "signal_period": input_values[1],
            }

        indicators_data.append({
            "type": indicator_type,
            "settings": settings
        })

    # Save the strategy to your database.
    success = save_strategy_to_db(strategy_id, strategy_name, risk_return_preference, indicators_data)

    if success:
        return [html.I(className="fas fa-check me-2"), "Saved"]
    else:
        return [html.I(className="fas fa-exclamation-triangle me-2"), "Error"]
        
        
@callback(
    Output('table-trigger', 'data'),
    [Input({'type': 'save-strategy-button', 'index': ALL}, 'n_clicks'),
     Input({'type': 'delete-strategy-button', 'index': ALL}, 'n_clicks')],
    State('table-trigger', 'data')
)
def update_trigger(save_clicks, delete_clicks, current_trigger):
    if dash.callback_context.triggered:
        triggered_id = dash.callback_context.triggered[0]['prop_id']
        if 'delete-strategy-button' in triggered_id:
            strategy_id = json.loads(triggered_id.split('.')[0])['index']
            delete_strategy_from_db(strategy_id)
    return (current_trigger or 0) + 1

@callback(
    Output("strategies-table-container", "children"),
    Input('table-trigger', 'data')
)
def update_strategies_table(trigger):
    strategies = load_strategies()
    if not strategies:
        return html.Div("No strategies saved yet.", className="mt-3")
    
    table = dbc.Table([
        html.Thead([
            html.Tr([
                html.Th("Strategy Name"),
                html.Th("Risk-Return Preference"),
                html.Th("Number of Indicators"),
                html.Th("Actions")
            ])
        ]),
        html.Tbody([
            html.Tr([
                html.Td(strategy["name"] or "Unnamed Strategy"),
                html.Td(f"{strategy['risk_return_preference']:.1f}"),
                html.Td(len(strategy["indicators"])),
                html.Td(
                    dbc.Button(
                        [html.I(className="fas fa-trash me-2"), "Delete"],
                        id={'type': 'delete-strategy-button', 'index': strategy["id"]},
                        color="danger",
                        size="sm"
                    )
                )
            ]) for strategy in strategies
        ])
    ], bordered=True, dark=True, hover=True, className="mt-3")
    
    return table

# Run the server
if __name__ == '__main__':
    app.run_server(debug=True, port=3000)