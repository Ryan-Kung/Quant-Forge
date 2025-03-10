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


# Helper Functions
def create_indicator_item(indicator_id):
    """Create an indicator selection box"""
    return html.Div([
        dbc.Select(
            id={
                'type': 'indicator-select',
                'index': indicator_id
            },
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
        html.Div(id={
            'type': 'indicator-config',
            'index': indicator_id
        })
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
                    html.H5("Risk-Return Preference"),
                    dbc.Row([
                        dbc.Col(
                            dbc.Label("Set your risk-return preference (higher values = higher risk tolerance):", 
                                     className="mb-2"),
                            width=12
                        ),
                        dbc.Col(
                            dbc.Input(
                                id={'type': 'risk-return-input', 'index': strategy_id},
                                type="number",
                                min=0.1,
                                max=10,
                                step=0.1,
                                value=risk_return_preference,
                                className="mb-3"
                            ),
                            width=6
                        )
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






import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
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


# Function to generate the Bokeh plot and embed it inside Dash
# Modify this function in your code:
import pandas as pd
import numpy as np
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, CrosshairTool, Legend
from bokeh.layouts import column
from bokeh.embed import file_html
from bokeh.resources import CDN

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
    
    # Create a stats table
    stats_tbl = figure(
        title="Backtest Statistics",
        width=1000,
        height=300,
        x_range=stats_data['Metric'],
        tools="",
        toolbar_location=None
    )
    
    # Remove grid lines and axis ticks
    stats_tbl.xgrid.grid_line_color = None
    stats_tbl.ygrid.grid_line_color = None
    stats_tbl.yaxis.visible = False
    
    # Add text for stats values
    y_pos = 0.5
    for i, (metric, value) in enumerate(zip(stats_data['Metric'], stats_data['Value'])):
        stats_tbl.text(
            x=metric, 
            y=y_pos, 
            text=[value],
            text_align="center",
            text_baseline="middle",
            text_font_size="14px"
        )
    
    # Layout
    layout = column(p, stats_tbl)
    
    # Convert to HTML
    html_content = file_html(layout, CDN, "Backtest Results")
    
    return html_content
# Historical Data Layout
historical_layout = html.Div([
    html.H2("Historical Data Backtesting", className="mt-3 mb-4"),

    # **TIDIED-UP INPUT SECTION**
    dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col(dbc.Input(id="ticker-input", type="text", placeholder="Enter Ticker (e.g., GOOG)", className="mb-2"), width=2),
                dbc.Col(dcc.Dropdown(
                    id="timeframe-dropdown",
                    options=[
                        {"label": "1 Minute", "value": "1m"},
                        {"label": "5 Minutes", "value": "5m"},
                        {"label": "15 Minutes", "value": "15m"},
                        {"label": "1 Hour", "value": "1h"},
                        {"label": "1 Day", "value": "1d"},
                    ],
                    placeholder="Select Timeframe",
                    className="mb-2",
                    style={'color': 'black'}
                ), width=2),
                dbc.Col(dcc.DatePickerSingle(id="start-date", placeholder="Start Date"), width=2),
                dbc.Col(dcc.DatePickerSingle(id="end-date", placeholder="End Date"), width=2),
                dbc.Col(dcc.Dropdown(id="strategy-dropdown", placeholder="Select Strategy", style={'color': 'black'}), width=2),
                dbc.Col(dbc.Input(id="cash-input", type="number", placeholder="Broker Cash (e.g., $10,000)", value=10000, className="mb-2"), width=2),
            ], className="g-2"),  # Adds spacing
            dbc.Row([
                dbc.Col(dbc.Button("Run Backtest", id="run-backtest-btn", color="primary", className="mt-2"), width="auto"),
            ], className="g-2"),
        ])
    ], className="mb-3"),

    # **BACKTEST PLOT**
    dbc.Row([
        dbc.Col(html.Iframe(id="backtest-bokeh-frame", style={"width": "100%", "height": "600px", "border": "none"}), width=12)
    ])
])

@dash.callback(
    Output("backtest-bokeh-frame", "srcDoc"),
    Input("run-backtest-btn", "n_clicks"),
    State("cash-input", "value"),
    prevent_initial_call=True
)
def update_backtest_graph(n_clicks, cash):
    try:
        # Use our custom plotting function instead of the built-in one
        return generate_custom_backtest_plot(cash)
    except Exception as e:
        # Create a simple error message figure as fallback
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
     Input("interval-component", "n_intervals")],  # Add an interval component for auto-refresh
    [State("alpaca-api-key", "value"),
     State("alpaca-api-secret", "value")],
    prevent_initial_call=True
)
def update_orders_table(refresh_clicks, start_clicks, interval, api_key, api_secret):
    # Use default keys if none provided
    api_key = api_key or default_key
    api_secret = api_secret or default_secret_key
    
    if not api_key or not api_secret:
        return [html.Tr([html.Td(colSpan=4, className="text-center"), 
                       html.Td("API credentials required to fetch orders.")])]
    
    try:
        # Initialize API
        api = tradeapi.REST(api_key, api_secret, BASE_URL, api_version='v2')
        
        # Get orders from the last 7 days (only filled ones)
        import datetime
        seven_days_ago = (datetime.datetime.now() - datetime.timedelta(days=7)).strftime('%Y-%m-%d')
        all_orders = api.list_orders(status='all', limit=100, after=seven_days_ago)
        
        # Filter for only filled orders
        filled_orders = [order for order in all_orders if order.status == 'filled']
        
        if not filled_orders:
            return [html.Tr([html.Td(colSpan=4, className="text-center"), 
                           html.Td("No filled orders found.")])]
        
        # Format orders for display - simplified version
        order_rows = []
        for order in filled_orders:
            # Apply color based on side
            side_color = "text-success" if order.side == 'buy' else "text-danger"
            
            # Format price with dollar sign
            price = f"${float(order.filled_avg_price):.2f}" if hasattr(order, 'filled_avg_price') and order.filled_avg_price else "-"
            
            # Format the filled time to be more readable
            filled_time = order.filled_at.replace('T', ' ').split('.')[0] if order.filled_at else "-"
            
            order_rows.append(html.Tr([
                html.Td(order.symbol),
                html.Td(order.side.upper(), className=side_color),
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
@callback(
    [Output("start-trading-btn", "disabled"),
     Output("stop-trading-btn", "disabled"),
     Output("trading-status", "children"),
     Output("trading-status", "className")],
    [Input("start-trading-btn", "n_clicks"),
     Input("stop-trading-btn", "n_clicks")],
    [State("live-strategy-dropdown", "value"),   
     State("symbol-input", "value"),
     State("alpaca-api-key", "value"),          
     State("alpaca-api-secret", "value")],
    prevent_initial_call=True
)
def handle_trading_actions(start_clicks, stop_clicks, strategy_id, symbol, alpaca_key, secret_key):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    thread_id = f"{strategy_id}_{symbol}" if strategy_id and symbol else None
    
    if button_id == "start-trading-btn":
        if not strategy_id or not symbol:
            return False, True, "Please select a strategy and symbol", "text-warning"
            
        # Use provided API keys or fall back to defaults if empty
        api_key = default_key
        api_secret = default_secret_key
        
        if not api_key or not api_secret:
            return False, True, "API credentials required", "text-warning"
            
        logger.info(f"Starting trading with strategy {strategy_id} for symbol {symbol}")
        
        # If already running, stop it first
        if thread_id in trading_threads:
            trading_threads[thread_id]["stop"] = True
            # Give it a moment to clean up
            time.sleep(1)
        
        # Start trading in a new thread
        import threading
        trading_thread = threading.Thread(
            target=run_live_trading,
            args=(symbol, strategy_id, api_key, api_secret, True)
        )
        trading_thread.daemon = True
        trading_thread.start()
        
        return True, False, f"Trading active: {symbol}", "text-success"
    
    elif button_id == "stop-trading-btn":
        # Signal thread to stop if it exists
        if thread_id in trading_threads:
            trading_threads[thread_id]["stop"] = True
            logger.info(f"Stopping trading for {symbol} with strategy {strategy_id}")
            return False, True, "Inactive", "text-warning"
        else:
            # No thread was running
            return False, True, "No active trading to stop", "text-warning"
    
    # Default - no change
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update


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