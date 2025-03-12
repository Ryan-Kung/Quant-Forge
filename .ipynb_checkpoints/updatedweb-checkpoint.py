#!/usr/bin/env python
# coding: utf-8

# In[117]:


import dash
from dash import dcc, html, Input, Output, State, callback, ALL, MATCH
import dash_bootstrap_components as dbc
import uuid
import sqlite3
import json
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA, GOOG
from bokeh.embed import file_html
from bokeh.resources import CDN
import io
import base64
import matplotlib.pyplot as plt

# Initialize the app with dark theme - using standard Dash
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
    
    # Get the current indicator index
    ctx = dash.callback_context
    indicator_id = ctx.outputs_list[0]['id']['index']
    
    if indicator_type == "RSI":
        return html.Div([
            dbc.Label("Period"),
            dbc.Input(type="number", value=14, min=1, max=100, id={'type': 'RSI-period', 'index': indicator_id}),
            dbc.Label("Overbought Level"),
            dbc.Input(type="number", value=70, min=1, max=100, id={'type': 'RSI-overbought', 'index': indicator_id}),
            dbc.Label("Oversold Level"),
            dbc.Input(type="number", value=30, min=1, max=100, id={'type': 'RSI-oversold', 'index': indicator_id}),
        ])
    elif indicator_type == "MACD":
        return html.Div([
            dbc.Label("Fast Period"),
            dbc.Input(type="number", value=12, min=1, max=100, id={'type': 'MACD-fast', 'index': indicator_id}),
            dbc.Label("Slow Period"),
            dbc.Input(type="number", value=26, min=1, max=100, id={'type': 'MACD-slow', 'index': indicator_id}),
            dbc.Label("Signal Period"),
            dbc.Input(type="number", value=9, min=1, max=100, id={'type': 'MACD-signal', 'index': indicator_id}),
        ])
    elif indicator_type == "STOCH":
        return html.Div([
            dbc.Label("K Period"),
            dbc.Input(type="number", value=14, min=1, max=100, id={'type': 'STOCH-k', 'index': indicator_id}),
            dbc.Label("D Period"),
            dbc.Input(type="number", value=3, min=1, max=100, id={'type': 'STOCH-d', 'index': indicator_id}),
            dbc.Label("Slow Period"),
            dbc.Input(type="number", value=3, min=1, max=100, id={'type': 'STOCH-slow', 'index': indicator_id}),
        ])
    elif indicator_type == "SMA_CROSS":
        return html.Div([
            dbc.Label("Fast Period"),
            dbc.Input(type="number", value=10, min=1, max=100, id={'type': 'SMA-fast', 'index': indicator_id}),
            dbc.Label("Slow Period"),
            dbc.Input(type="number", value=20, min=1, max=100, id={'type': 'SMA-slow', 'index': indicator_id}),
        ])
    elif indicator_type == "BBANDS":
        return html.Div([
            dbc.Label("Period"),
            dbc.Input(type="number", value=20, min=1, max=100, id={'type': 'BBANDS-period', 'index': indicator_id}),
            dbc.Label("Standard Deviation"),
            dbc.Input(type="number", value=2, min=0.1, max=10, step=0.1, id={'type': 'BBANDS-std-dev', 'index': indicator_id}),
        ])
    elif indicator_type == "SMI":
        return html.Div([
            dbc.Label("Period"),
            dbc.Input(type="number", value=14, min=1, max=100, id={'type': 'SMI-period', 'index': indicator_id}),
            dbc.Label("Signal Period"),
            dbc.Input(type="number", value=9, min=1, max=100, id={'type': 'SMI-signal', 'index': indicator_id}),
        ])
    return html.Div("Select an indicator to configure its parameters")

# Fixed: Use separate callbacks for saving and table trigger updates
@callback(
    Output({'type': 'save-strategy-button', 'index': MATCH}, "children"),
    Input({'type': 'save-strategy-button', 'index': MATCH}, "n_clicks"),
    [State({'type': 'strategy-name-input', 'index': MATCH}, "value"),
     State({'type': 'risk-return-input', 'index': MATCH}, "value"),
     State({'type': 'indicators-container', 'index': MATCH}, "children")],
    prevent_initial_call=True
)
def handle_save_button(n_clicks, strategy_name, risk_return_preference, indicators_children):
    if not n_clicks:
        return dash.no_update

    # Get strategy_id from callback context
    ctx = dash.callback_context
    triggered_prop = ctx.triggered[0]['prop_id']
    try:
        strategy_id = json.loads(triggered_prop.split('.')[0])['index']
    except Exception as e:
        print("Error parsing strategy id:", e)
        return [html.I(className="fas fa-exclamation-triangle me-2"), "Error"]

    # Extract indicator data (simplified for demonstration)
    indicators_data = []
    for child in indicators_children or []:
        indicators_data.append({
            "type": "placeholder",
            "settings": {}
        })

    # Save the strategy to the database
    success = save_strategy_to_db(strategy_id, strategy_name, risk_return_preference, indicators_data)
    
    # Update the table trigger in a separate callback
    if success:
        # Trigger table update using clientside callback in a real app
        return [html.I(className="fas fa-check me-2"), "Saved"]
    else:
        return [html.I(className="fas fa-exclamation-triangle me-2"), "Error"]

# Fixed: Separate callback for table trigger updates on save
@callback(
    Output('table-trigger', 'data', allow_duplicate=True),
    Input({'type': 'save-strategy-button', 'index': ALL}, 'n_clicks'),
    State('table-trigger', 'data'),
    prevent_initial_call=True
)
def update_table_on_save(save_clicks, current_trigger):
    # Only update if any button was clicked
    if any(click is not None for click in save_clicks):
        return (current_trigger or 0) + 1
    return dash.no_update

# Fixed: Separate callback for table trigger updates on delete
@callback(
    Output('table-trigger', 'data', allow_duplicate=True),
    Input({'type': 'delete-strategy-button', 'index': ALL}, 'n_clicks'),
    State('table-trigger', 'data'),
    prevent_initial_call=True
)
def update_table_on_delete(delete_clicks, current_trigger):
    ctx = dash.callback_context
    if ctx.triggered:
        triggered_id = ctx.triggered[0]['prop_id']
        if 'delete-strategy-button' in triggered_id:
            strategy_id = json.loads(triggered_id.split('.')[0])['index']
            delete_strategy_from_db(strategy_id)
            return (current_trigger or 0) + 1
    return dash.no_update

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

# Callback to update backtest graph using Bokeh inside Dash
@callback(
    Output("backtest-bokeh-frame", "srcDoc"),
    Input("run-backtest-btn", "n_clicks"),
    State("cash-input", "value"),
    prevent_initial_call=True
)
def update_backtest_graph(n_clicks, cash):
    return generate_backtest_bokeh(cash)

# Callback to update Strategy Dropdown
@callback(
    Output("strategy-dropdown", "options"),
    Input("table-trigger", "data")
)
def update_strategy_dropdown(trigger):
    strategies = load_strategies()
    return [{"label": s["name"], "value": s["id"]} for s in strategies] if strategies else []

# Run the app
if __name__ == '__main__':
    # For standard Dash, use the following
    app.run_server(debug=True, port=10000)

