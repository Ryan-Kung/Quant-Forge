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
                indicators TEXT
            )
        """)
        strategy_db.commit()
        return strategy_db

def save_strategy_to_db(strategy_id, strategy_name, indicators_data):
    """Save strategy to database."""
    try:
        db = get_strategy_db()
        cursor = db.cursor()
        indicators_json = json.dumps(indicators_data)
        
        cursor.execute("SELECT id FROM strategy WHERE id = ?", (strategy_id,))
        if cursor.fetchone():
            cursor.execute(
                "UPDATE strategy SET name = ?, indicators = ? WHERE id = ?",
                (strategy_name, indicators_json, strategy_id)
            )
        else:
            cursor.execute(
                "INSERT INTO strategy (id, name, indicators) VALUES (?, ?, ?)",
                (strategy_id, strategy_name, indicators_json)
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
    cursor.execute("SELECT id, name, indicators FROM strategy")
    strategies = []
    
    for row in cursor.fetchall():
        strategy_id, name, indicators_json = row
        indicators = json.loads(indicators_json) if indicators_json else []
        strategies.append({
            "id": strategy_id,
            "name": name,
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

def create_strategy_card(strategy_id, strategy_name=None):
    """Create a strategy card with name editing and save functionality."""
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
                    html.H5("Indicators"),
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

historical_layout = html.Div([
    html.H2("Historical Data", className="mt-3"),
    html.P("Historical data content will go here.")
])

live_layout = html.Div([
    html.H2("Live Trading", className="mt-3"),
    html.P("Live trading content will go here.")
])

# Main layout
app.layout = html.Div([
    dbc.Container([
        html.H1("Trader", className="mt-3 mb-3"),
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
    if indicator_type == "RSI":
        return html.Div([
            dbc.Label("Period"),
            dbc.Input(type="number", value=14, min=1, max=100),
            dbc.Label("Overbought Level"),
            dbc.Input(type="number", value=70, min=1, max=100),
            dbc.Label("Oversold Level"),
            dbc.Input(type="number", value=30, min=1, max=100),
        ])
    elif indicator_type == "MACD":
        return html.Div([
            dbc.Label("Fast Period"),
            dbc.Input(type="number", value=12, min=1, max=100),
            dbc.Label("Slow Period"),
            dbc.Input(type="number", value=26, min=1, max=100),
            dbc.Label("Signal Period"),
            dbc.Input(type="number", value=9, min=1, max=100),
        ])
    elif indicator_type == "STOCH":
        return html.Div([
            dbc.Label("K Period"),
            dbc.Input(type="number", value=14, min=1, max=100),
            dbc.Label("D Period"),
            dbc.Input(type="number", value=3, min=1, max=100),
            dbc.Label("Slow Period"),
            dbc.Input(type="number", value=3, min=1, max=100),
        ])
    elif indicator_type == "SMA_CROSS":
        return html.Div([
            dbc.Label("Fast SMA Period"),
            dbc.Input(type="number", value=10, min=1, max=200),
            dbc.Label("Slow SMA Period"),
            dbc.Input(type="number", value=20, min=1, max=200),
        ])
    elif indicator_type == "BBANDS":
        return html.Div([
            dbc.Label("Period"),
            dbc.Input(type="number", value=20, min=1, max=100),
            dbc.Label("Standard Deviation"),
            dbc.Input(type="number", value=2, min=1, max=10),
        ])
    elif indicator_type == "SMI":
        return html.Div([
            dbc.Label("Period"),
            dbc.Input(type="number", value=14, min=1, max=100),
            dbc.Label("Signal Period"),
            dbc.Input(type="number", value=9, min=1, max=100),
        ])
    return html.Div("Select an indicator to configure its parameters")

@callback(
    Output({'type': 'save-strategy-button', 'index': MATCH}, "children"),
    Input({'type': 'save-strategy-button', 'index': MATCH}, "n_clicks"),
    [State({'type': 'strategy-name-input', 'index': MATCH}, "value"),
     State({'type': 'indicators-container', 'index': MATCH}, "children")],
    prevent_initial_call=True
)
def handle_save_button(n_clicks, strategy_name, indicators):
    if n_clicks:
        triggered_id = dash.callback_context.triggered[0]['prop_id']
        strategy_id = json.loads(triggered_id.split('.')[0])['index']
        
        indicators_data = []
        for indicator in indicators or []:
            indicators_data.append({
                "type": "placeholder",
                "settings": {}
            })
        
        save_strategy_to_db(strategy_id, strategy_name, indicators_data)
        return [html.I(className="fas fa-save me-2"), "Save"]
    return dash.no_update

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
                html.Th("Number of Indicators"),
                html.Th("Actions")
            ])
        ]),
        html.Tbody([
            html.Tr([
                html.Td(strategy["name"] or "Unnamed Strategy"),
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