import json
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import logging
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import matplotlib.pyplot as plt
from bokeh.embed import file_html
from bokeh.resources import CDN

# Set up logging for this example
logger = logging.getLogger('backtest')
logger.setLevel(logging.ERROR)  # Changed from INFO to ERROR to minimize logging
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)  # Changed from INFO to ERROR
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# -------------------------------
# JSON Strategy Parser Function
# -------------------------------
def parse_strategy_json(strategy_json):
    """
    Parse a strategy JSON string or dict into the format needed for the backtesting system.
    
    Parameters:
    - strategy_json: String or dict containing the strategy configuration
    
    Returns:
    - A tuple containing (sma_settings, rsi_settings, macd_settings, risk_return)
    """
    # Convert string to dict if necessary
    if isinstance(strategy_json, str):
        strategy_dict = json.loads(strategy_json)
    else:
        strategy_dict = strategy_json
    
    # Initialize empty settings
    sma_settings = []
    rsi_settings = []
    macd_settings = []
    risk_return = strategy_dict.get('Risk-Return Preference', 1.0)
    
    # Parse indicators
    indicators = strategy_dict.get('Indicators', [])
    for indicator in indicators:
        indicator_type = indicator.get('type')
        settings = indicator.get('settings', {})
        
        if indicator_type == 'RSI':
            period = settings.get('period', 14)
            overbought = settings.get('overbought', 70)
            oversold = settings.get('oversold', 30)
            rsi_settings = [period, overbought, oversold]
            
        elif indicator_type == 'MACD':
            fast_period = settings.get('fast_period', 12)
            slow_period = settings.get('slow_period', 26)
            signal_period = settings.get('signal_period', 9)
            macd_settings = [fast_period, slow_period, signal_period]
            
        elif indicator_type == 'SMA_CROSS':
            fast_period = settings.get('fast_period', 10)
            slow_period = settings.get('slow_period', 20)
            sma_settings = [fast_period, slow_period]
    
    return sma_settings, rsi_settings, macd_settings, risk_return

# -------------------------------
# Historical Data Retrieval
# -------------------------------
def get_historical_data(symbol, start, end, interval='1d'):
    """
    Retrieve historical OHLC data using yfinance.
    Returns a DataFrame with columns: ['open', 'high', 'low', 'close']
    """
    try:
        df = yf.download(symbol, start=start, end=end, interval=interval)
        if df.empty:
            logger.warning(f"No historical data fetched for symbol {symbol}")
            return pd.DataFrame(columns=['open', 'high', 'low', 'close'])
        # Select necessary columns and rename to lowercase
        df = df[['Open', 'High', 'Low', 'Close']].copy()
        df.columns = ['open', 'high', 'low', 'close']
        return df
    except Exception as e:
        logger.error(f"Error fetching historical data from yfinance: {e}")
        return pd.DataFrame(columns=['open', 'high', 'low', 'close'])

# -------------------------------
# Indicator Calculation Function
# -------------------------------
def calculate_indicators(df, sma_settings, rsi_settings, macd_settings):
    """
    Calculate technical indicators based on provided settings.
    
    Parameters:
    - df: DataFrame with OHLC data
    - sma_settings: List containing [fast_period, slow_period] for SMA crossover
    - rsi_settings: List containing [period, overbought, oversold] for RSI
    - macd_settings: List containing [fast_period, slow_period, signal_period] for MACD
    
    Returns:
    - DataFrame with added indicator columns
    """
    if df.empty:
        logger.warning("Empty DataFrame, skipping indicator calculation.")
        return df
        
    result_df = df.copy()
    
    # SMA calculation (if SMA settings provided)
    if sma_settings and len(sma_settings) >= 2:
        result_df['SMA_fast'] = ta.sma(result_df['close'], length=sma_settings[0])
        result_df['SMA_slow'] = ta.sma(result_df['close'], length=sma_settings[1])
    
    # RSI calculation
    if rsi_settings and len(rsi_settings) >= 1:
        result_df['RSI'] = ta.rsi(result_df['close'], length=rsi_settings[0])
    
    # MACD calculation
    if macd_settings and len(macd_settings) >= 3:
        fast, slow, signal = macd_settings
        macd_df = ta.macd(
            result_df['close'], 
            fast=fast, 
            slow=slow, 
            signal=signal
        )
        for col in macd_df.columns:
            result_df[col] = macd_df[col]
    
    return result_df

# -------------------------------
# Custom Strategy Factory
# -------------------------------
def create_custom_strategy(risk_return, sma_settings, rsi_settings, macd_settings):
    """
    Create a custom strategy class using the provided indicator settings.
    
    Parameters:
    - risk_return: The risk/reward ratio for exits
    - sma_settings: List with [fast_period, slow_period] for SMA crossover
    - rsi_settings: List with [period, overbought, oversold] for RSI
    - macd_settings: List with [fast_period, slow_period, signal_period] for MACD
    
    Returns:
    - A Strategy class that can be used with backtesting.py
    """
    class CustomStrategy(Strategy):
        def init(self):
            self.risk_return = risk_return
            
            # Initialize indicators using self.I to properly integrate with backtesting.py
            
            # 1. SMA indicators
            if 'SMA_fast' in self.data.df.columns and 'SMA_slow' in self.data.df.columns:
                self.sma_fast = self.I(lambda: self.data.df['SMA_fast'])
                self.sma_slow = self.I(lambda: self.data.df['SMA_slow'])
                self.has_sma = True
            else:
                self.has_sma = False
            
            # 2. RSI indicator
            if 'RSI' in self.data.df.columns:
                self.rsi = self.I(lambda: self.data.df['RSI'])
                self.rsi_period = rsi_settings[0] if rsi_settings else 14
                self.rsi_overbought = rsi_settings[1] if len(rsi_settings) > 1 else 70
                self.rsi_oversold = rsi_settings[2] if len(rsi_settings) > 2 else 30
                self.has_rsi = True
            else:
                self.has_rsi = False
            
            # 3. MACD indicators
            macd_cols = [col for col in self.data.df.columns if col.startswith('MACD')]
            if len(macd_cols) >= 2:
                self.macd = self.I(lambda: self.data.df[macd_cols[0]])
                self.macd_signal = self.I(lambda: self.data.df[macd_cols[1]])
                if len(macd_cols) > 2:
                    self.macd_hist = self.I(lambda: self.data.df[macd_cols[2]])
                self.macd_fast = macd_settings[0] if macd_settings else 12
                self.macd_slow = macd_settings[1] if len(macd_settings) > 1 else 26
                self.macd_signal_period = macd_settings[2] if len(macd_settings) > 2 else 9
                self.has_macd = True
            else:
                self.has_macd = False
            
            # Track entry price for position management
            self.entry_price = None

        def check_sma_condition(self):
            """SMA condition: Fast SMA is above Slow SMA or crossing above"""
            if self.has_sma:
                # Either fast SMA is already above slow SMA
                fast_above_slow = self.sma_fast[-1] > self.sma_slow[-1]
                # Or fast SMA is crossing above slow SMA
                crossing_above = (self.sma_fast[-1] > self.sma_slow[-1]) and (self.sma_fast[-2] <= self.sma_slow[-2] if len(self.sma_fast) > 1 else False)
                return fast_above_slow or crossing_above
            return False
        
        def check_rsi_condition(self):
            """RSI condition: RSI is below 45 (more lenient than standard oversold)"""
            if self.has_rsi:
                # More lenient RSI condition - below 45 instead of 30
                return self.rsi[-1] < 45
            return False
        
        def check_macd_condition(self):
            """MACD condition: MACD is above signal line or crossing above"""
            if self.has_macd:
                # Either MACD is already above signal line
                macd_above_signal = self.macd[-1] > self.macd_signal[-1]
                # Or MACD is crossing above signal line
                macd_crossing = (self.macd[-1] > self.macd_signal[-1]) and (self.macd[-2] <= self.macd_signal[-2] if len(self.macd) > 1 else False)
                return macd_above_signal or macd_crossing
            return False

        def buy_condition(self):
            """
            Combined buy condition checking all available indicators.
            At least 2 indicators must signal a buy for a trade to be executed.
            """
            # Store condition results
            conditions = {}
            
            if self.has_sma:
                conditions["SMA"] = self.check_sma_condition()
            
            if self.has_rsi:
                conditions["RSI"] = self.check_rsi_condition()
            
            if self.has_macd:
                conditions["MACD"] = self.check_macd_condition()
            
            # Require at least 2 indicators to agree
            true_conditions = sum(conditions.values())
            min_required = min(2, len(conditions))  # At least 2 or all if less than 2 indicators
            
            return true_conditions >= min_required

        def next(self):
            """
            Main strategy method called for each bar in the data.
            Handles entry and exit decisions.
            """
            # If not in a position, check buy conditions
            if not self.position:
                if self.buy_condition():
                    self.buy()
                    self.entry_price = self.data.Close[-1]
            else:
                # Handle existing position
                if self.entry_price is None:
                    self.entry_price = self.data.Close[-1]
                    
                current_price = self.data.Close[-1]
                pct_change = (current_price - self.entry_price) / self.entry_price * 100
                
                # Set a base risk percentage (the amount we're willing to lose)
                base_risk_pct = 1.0  # 1% base risk
                
                # Calculate profit target based on risk-return ratio
                # If risk-return is 3, we want to risk 1% to make 3%
                tp = self.risk_return * base_risk_pct  # Take profit threshold percentage
                sl = -base_risk_pct  # Stop loss threshold percentage always fixed at base_risk_pct
                
                # Check for take profit or stop loss
                if pct_change >= tp:
                    self.position.close()
                    self.entry_price = None
                elif pct_change <= sl:
                    self.position.close()
                    self.entry_price = None
                    
    return CustomStrategy

# -------------------------------
# Backtest Runner Function
# -------------------------------
def run_backtest(symbol, start, end, strategy_json, cash=10000, interval='1d'):
    """
    Run a backtest using the provided strategy JSON.
    
    Parameters:
    - symbol: Stock symbol to backtest on
    - start: Start date for historical data
    - end: End date for historical data
    - strategy_json: Strategy configuration in JSON format
    - cash: Initial capital for backtest
    - interval: Data interval (e.g. '1d' for daily)
    
    Returns:
    - A tuple containing (html_content, stats)
    """
    try:
        # Parse strategy settings
        sma_settings, rsi_settings, macd_settings, risk_return = parse_strategy_json(strategy_json)
        
        # Get historical data
        df = get_historical_data(symbol, start, end, interval=interval)
        if df.empty:
            logger.error("No historical data available.")
            from bokeh.plotting import figure
            from bokeh.embed import file_html
            from bokeh.resources import CDN
            
            error_fig = figure(title="No Historical Data", width=1000, height=300)
            error_fig.text(x=0.5, y=0.5, text=["No historical data available for the selected symbol and date range."],
                          text_align="center", text_baseline="middle", text_font_size="14px")
            # Return empty stats dict with the error
            empty_stats = {
                'Equity Final [$]': 0, 
                'Return [%]': 0, 
                'Max. Drawdown [%]': 0, 
                '# Trades': 0, 
                'Win Rate [%]': 0,
                'Best Trade [%]': 0,
                'Worst Trade [%]': 0
            }
            return file_html(error_fig, CDN), empty_stats
        
        # Calculate indicators
        df_with_indicators = calculate_indicators(df, sma_settings, rsi_settings, macd_settings)
        
        # Rename price columns to match backtesting.py's expected names
        df_with_indicators.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)
        
        # Create the strategy
        StrategyClass = create_custom_strategy(risk_return, sma_settings, rsi_settings, macd_settings)
        
        # Initialize and run the backtest
        bt = Backtest(df_with_indicators, StrategyClass, cash=cash, commission=0.002, exclusive_orders=True)
        stats = bt.run()
        
        # Print summary statistics for console/logs
        print("\nBacktest Summary:")
        print(f"Start Date: {start}, End Date: {end}, Symbol: {symbol}")
        print(f"Initial Capital: ${cash:.2f}")
        print(f"Final Portfolio Value: ${stats['Equity Final [$]']:.2f}")
        print(f"Total Return: {stats['Return [%]']:.2f}%")
        print(f"Max Drawdown: {stats['Max. Drawdown [%]']:.2f}%")
        print(f"Win Rate: {stats['Win Rate [%]']:.2f}%")
        print(f"Total Number of Trades: {stats['# Trades']}")
        
        # Try to generate the HTML using built-in plot function
        try:
            # First try with open_browser=False
            html_content = bt.plot(open_browser=False)
            return html_content, stats
            
        except TypeError:
            try:
                # If that fails, try with no parameters
                html_content = bt.plot()
                return html_content, stats
                
            except Exception as plot_error:
                logger.error(f"Error in bt.plot(): {str(plot_error)}")
                
                # Create a fallback error figure
                from bokeh.plotting import figure
                from bokeh.embed import file_html
                from bokeh.resources import CDN
                
                error_fig = figure(title=f"Error Generating Plot", width=1000, height=300)
                error_message = f"Could not generate plot: {str(plot_error)}"
                error_fig.text(x=0.5, y=0.5, text=[error_message],
                              text_align="center", text_baseline="middle", text_font_size="14px")
                
                # Still return the stats along with the error figure
                return file_html(error_fig, CDN), stats
        
    except Exception as e:
        import traceback
        logger.error(f"Error in backtest: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Create error message figure
        from bokeh.plotting import figure
        from bokeh.embed import file_html
        from bokeh.resources import CDN
        
        error_fig = figure(title=f"Error Running Backtest", width=1000, height=300)
        error_message = f"An error occurred: {str(e)}"
        error_fig.text(x=0.5, y=0.5, text=[error_message],
                      text_align="center", text_baseline="middle", text_font_size="14px")
        
        # Return empty stats dict with the error
        empty_stats = {
            'Equity Final [$]': 0, 
            'Return [%]': 0, 
            'Max. Drawdown [%]': 0, 
            '# Trades': 0, 
            'Win Rate [%]': 0,
            'Best Trade [%]': 0,
            'Worst Trade [%]': 0
        }
        
        return file_html(error_fig, CDN), empty_stats

# -------------------------------
# Example Usage
# -------------------------------
if __name__ == "__main__":
    # Example strategy JSON - use your specific strategy
    strategy_json = """
    {
        "ID": "b3cc3f6b",
        "Risk-Return Preference": 2.0,
        "Number of Indicators": 3,
        "Indicators": [
            {
                "type": "RSI",
                "settings": {
                    "period": 14,
                    "overbought": 70,
                    "oversold": 30
                }
            },
            {
                "type": "MACD",
                "settings": {
                    "fast_period": 12,
                    "slow_period": 26,
                    "signal_period": 9
                }
            },
            {
                "type": "SMA_CROSS",
                "settings": {
                    "fast_period": 10,
                    "slow_period": 20
                }
            }
        ]
    }
    """
    
    # Define backtest parameters
    symbol = "SPY"
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    initial_cash = 10000
    
    # Run the backtest
    html_result = run_backtest(symbol, start_date, end_date, strategy_json, cash=initial_cash, interval='1d')
    
    # In a real app, this HTML would be displayed in the iframe
    print(f"Generated HTML result with {len(html_result)} characters")