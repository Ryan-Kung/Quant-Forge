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



#YOU CAN DELETE THIS BOTTOM PART
# def update_strategy_db():
#     """Updates the strategy table to include position_size if it doesn't exist."""
#     db = get_strategy_db()
#     cursor = db.cursor()
    
#     # Check if column exists before altering table
#     cursor.execute("PRAGMA table_info(strategy)")
#     columns = [col[1] for col in cursor.fetchall()]
    
#     if "position_size" not in columns:
#         cursor.execute("ALTER TABLE strategy ADD COLUMN position_size REAL DEFAULT 0.3")
#         db.commit()

# # Run this once after updating the code
# try:
#     update_strategy_db()
# except sqlite3.OperationalError:
#     pass  # Ignore if column already exists


def save_strategy_to_db(strategy_id, strategy_name, risk_return_preference, position_size, indicators_data):
    """Save strategy including position size into the database."""
    try:
        db = get_strategy_db()
        cursor = db.cursor()
        indicators_json = json.dumps(indicators_data)

        cursor.execute("SELECT id FROM strategy WHERE id = ?", (strategy_id,))
        if cursor.fetchone():
            cursor.execute(
                "UPDATE strategy SET name = ?, risk_return_preference = ?, position_size = ?, indicators = ? WHERE id = ?",
                (strategy_name, risk_return_preference, position_size, indicators_json, strategy_id)
            )
        else:
            cursor.execute(
                "INSERT INTO strategy (id, name, risk_return_preference, position_size, indicators) VALUES (?, ?, ?, ?, ?)",
                (strategy_id, strategy_name, risk_return_preference, position_size, indicators_json)
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
def generate_backtest_bokeh(cash):
    bt = Backtest(GOOG, SmaCross, cash=cash, commission=.002, exclusive_orders=True)
    output = bt.run()
    bokeh_fig = bt.plot(resample=False)  # Generate Bokeh figure

    # Convert Bokeh plot to HTML
    html_content = file_html(bokeh_fig, CDN)

    return html_content


# Historical Data Layout
historical_layout = html.Div([
    html.H2("Historical Data Backtesting", className="mt-3 mb-4"),

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
                dbc.Col(dcc.DatePickerSingle(
                    id="start-date",
                    placeholder="Start Date",
                ), width=2),
                dbc.Col(dcc.DatePickerSingle(
                    id="end-date",
                    placeholder="End Date",
                ), width=2),
                dbc.Col(dcc.Dropdown(
                    id="strategy-dropdown",
                    placeholder="Select Strategy",
                    style={'color': 'black', "width": "100%", "height": "38px"}
                ), width=2),
                dbc.Col(dbc.Input(id="cash-input", type="number", placeholder="Broker Cash (e.g., $10,000)", value=10000, className="mb-2"), width=2),
            ], className="g-2"),
            dbc.Row([
                dbc.Col(dbc.Button("Run Backtest", id="run-backtest-btn", color="primary", className="mt-2"), width="auto"),
            ], className="g-2"),
        ])
    ], className="mb-3"),

    dbc.Row([
        dbc.Col(html.Iframe(id="backtest-bokeh-frame", style={"width": "100%", "height": "600px", "border": "none"}), width=12)
    ])
])



# Callback to update backtest graph using Bokeh inside Dash
@dash.callback(
    Output("backtest-bokeh-frame", "srcDoc"),
    Input("run-backtest-btn", "n_clicks"),
    State("cash-input", "value"),
    prevent_initial_call=True
)
def update_backtest_graph(n_clicks, cash):
    return generate_backtest_bokeh(cash)


# Callback to update Strategy Dropdown
@dash.callback(
    Output("strategy-dropdown", "options"),
    Input("table-trigger", "data")
)
def update_strategy_dropdown(trigger):
    strategies = load_strategies()
    return [{"label": s["name"], "value": s["id"]} for s in strategies] if strategies else []



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
     State({'type': 'position-size-input', 'index': MATCH}, "value"),  # Capture position size
     State({'type': 'indicator-select', 'index': ALL}, "value"),
     State({'type': 'indicator-select', 'index': ALL}, "id"),
     State({'type': 'indicator-config', 'index': ALL}, "children")],
    prevent_initial_call=True
)
def handle_save_button(n_clicks, strategy_name, risk_return_preference, position_size, indicator_types, indicator_ids, indicator_configs):
    """Handles saving the strategy, including Position Size."""

    # Extract input values from indicator configurations
    def extract_input_values(component):
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
                if child.get("type") == "Input":
                    val = child.get("props", {}).get("value")
                    if val is not None:
                        vals.append(val)
                else:
                    vals.extend(extract_input_values(child))
            elif isinstance(child, list):
                vals.extend(extract_input_values(child))
        return vals

    if not n_clicks:
        return dash.no_update

    ctx = dash.callback_context
    triggered_prop = ctx.triggered[0]['prop_id']
    try:
        strategy_id = json.loads(triggered_prop.split('.')[0])['index']
    except Exception as e:
        print("Error parsing strategy id:", e)
        return [html.I(className="fas fa-exclamation-triangle me-2"), "Error"]

    indicators_data = []
    for i, config in enumerate(indicator_configs):
        try:
            indicator_type = indicator_types[i]
        except IndexError:
            continue

        if not indicator_type:
            continue

        input_values = extract_input_values(config)

        settings = {}
        if indicator_type == "RSI" and len(input_values) >= 3:
            settings = {"period": input_values[0], "overbought": input_values[1], "oversold": input_values[2]}
        elif indicator_type == "MACD" and len(input_values) >= 3:
            settings = {"fast_period": input_values[0], "slow_period": input_values[1], "signal_period": input_values[2]}
        elif indicator_type == "STOCH" and len(input_values) >= 3:
            settings = {"k_period": input_values[0], "d_period": input_values[1], "slow_period": input_values[2]}
        elif indicator_type == "SMA_CROSS" and len(input_values) >= 2:
            settings = {"fast_period": input_values[0], "slow_period": input_values[1]}
        elif indicator_type == "BBANDS" and len(input_values) >= 2:
            settings = {"period": input_values[0], "std_dev": input_values[1]}
        elif indicator_type == "SMI" and len(input_values) >= 2:
            settings = {"period": input_values[0], "signal_period": input_values[1]}

        indicators_data.append({"type": indicator_type, "settings": settings})

    # Save the strategy with position size
    success = save_strategy_to_db(strategy_id, strategy_name, risk_return_preference, position_size, indicators_data)

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