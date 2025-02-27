import dash
from dash import dcc, html, Input, Output, State, callback, ALL, MATCH
import dash_bootstrap_components as dbc
import uuid
import sqlite3
import json

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

# Callback to update Strategy Dropdown
@dash.callback(
    Output("strategy-dropdown", "options"),
    Input("table-trigger", "data")
)
def update_strategy_dropdown(trigger):
    strategies = load_strategies()
    return [{"label": s["name"], "value": s["id"]} for s in strategies] if strategies else []










# historical_layout = html.Div([
#     html.H2("Historical Data", className="mt-3"),
#     html.P("Historical data content will go here.")
# ])

live_layout = html.Div([
    html.H2("Live Trading", className="mt-3"),
    html.P("Live trading content will go here.")
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

# Callbacks
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