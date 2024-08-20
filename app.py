import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from yahoo_fin import stock_info as si
import base64
import io

app = dash.Dash(__name__)
server = app.server  # This line is important for gunicorn to find the WSGI application

app.layout = html.Div([
    html.H1("TD Sequential"),
    html.Div([
        html.Label("Ticker:"),
        dcc.Input(id='ticker-input', value='AAPL', type='text')
    ]),
    html.Div([
        html.Label("Select Period:"),
        dcc.Dropdown(
            id='period-dropdown',
            options=[
                {'label': '3 Months', 'value': '3mo'},
                {'label': '6 Months', 'value': '6mo'},
                {'label': '1 Year', 'value': '1y'},
                {'label': 'YTD', 'value': 'ytd'},
                {'label': '5 Years', 'value': '5y'}
            ],
            value='ytd'
        )
    ]),
    html.Button('Submit', id='submit-button', n_clicks=0),
    html.Hr(),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=False
    ),
    html.Hr(),
    dcc.Graph(id='graph-output'),
    html.Div(id='countdown-table'),
    html.Div(id='aggressive-countdown-table'),
    html.Div([
        html.Label("Show Volume:"),
        dcc.RadioItems(
            id='volume-toggle',
            options=[
                {'label': 'On', 'value': 'on'},
                {'label': 'Off', 'value': 'off'}
            ],
            value='off',
            labelStyle={'display': 'inline-block'}
        )
    ]),
    html.Div([
        html.Label("Show Bollinger Bands:"),
        dcc.RadioItems(
            id='bollinger-toggle',
            options=[
                {'label': 'On', 'value': 'on'},
                {'label': 'Off', 'value': 'off'}
            ],
            value='off',
            labelStyle={'display': 'inline-block'}
        )
    ])
])

def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(f"Error parsing file {filename}: {e}")
        return pd.DataFrame()
    return df

def fetch_data(ticker, period):
    try:
        if period == '3mo':
            data = si.get_data(ticker, start_date=pd.to_datetime('today') - pd.DateOffset(months=3))
        elif period == '6mo':
            data = si.get_data(ticker, start_date=pd.to_datetime('today') - pd.DateOffset(months=6))
        elif period == '1y':
            data = si.get_data(ticker, start_date=pd.to_datetime('today') - pd.DateOffset(years=1))
        elif period == 'ytd':
            data = si.get_data(ticker, start_date=pd.to_datetime('today').replace(month=1, day=1))
        elif period == '5y':
            data = si.get_data(ticker, start_date=pd.to_datetime('today') - pd.DateOffset(years=5))
        else:
            data = si.get_data(ticker)
        
        if data.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        
        data.reset_index(inplace=True)
        data.rename(columns={'index': 'Date'}, inplace=True)
        return data
    except Exception as e:
        print(f"Error fetching data for ticker {ticker}: {e}")
        return pd.DataFrame()

def exponential_moving_average(values, window):
    ema = [0] * len(values)
    k = 2 / (window + 1)
    for i in range(len(values)):
        if i < window:
            ema[i] = sum(values[:i+1]) / (i+1)  # Use the average of available values for the initial period
        else:
            ema[i] = values[i] * k + ema[i-1] * (1 - k)
    return ema

def bollinger_bands(values, window, num_sd):
    sma = pd.Series(values).rolling(window=window, min_periods=1).mean()
    std = pd.Series(values).rolling(window=window, min_periods=1).std()
    upper_band = sma + (std * num_sd)
    lower_band = sma - (std * num_sd)
    upper_band.ffill(inplace=True)
    lower_band.ffill(inplace=True)
    return upper_band, lower_band

def apply_countdown_deferral(countdown_vals, highs, lows, closes):
    deferred = [False] * len(countdown_vals)
    for i in range(len(countdown_vals)):
        if countdown_vals[i] == 13:
            if i >= 12:
                if lows[i] > closes[i-8]:
                    deferred[i] = True
    return deferred

def check_recycling(countdown_vals, setups):
    recycled = [False] * len(countdown_vals)
    for i in range(len(countdown_vals)):
        if countdown_vals[i] == 13:
            for j in range(i, i-22, -1):
                if j >= 0 and setups[j] >= 9:
                    recycled[i] = True
                    break
    return recycled

def calculate_td_sequential(data):
    o = data['open'].tolist() if 'open' in data.columns else data['Open'].tolist()
    h = data['high'].tolist() if 'high' in data.columns else data['High'].tolist()
    l = data['low'].tolist() if 'low' in data.columns else data['Low'].tolist()
    c = data['close'].tolist() if 'close' in data.columns else data['Close'].tolist()
    t = pd.to_datetime(data['Date']).tolist()
    if 'volume' in data.columns:
        v = data['volume'].tolist()
    elif 'Volume' in data.columns:
        v = data['Volume'].tolist()
    else:
        v = [0] * len(c)  # Default volume to 0 if not present

    vol = [float(round(float(val) / (9**7), 3)) for val in v]

    shortVal, longVal, shortCount, longCount = [], [], 0, 0
    for k in range(len(c)):
        if k > 3:
            if c[k] < c[k-4]:
                longCount += 1
            else:
                longCount = 0
            if c[k] > c[k-4]:
                shortCount += 1
            else:
                shortCount = 0
        longVal.append(longCount)
        shortVal.append(shortCount)

    buyVal, sellVal, buyCount, sellCount = [], [], 0, 0
    for y in range(len(c)):
        if y >= 11:
            if buyCount == 0 and (h[y] >= l[y-3] or h[y] >= l[y-4] or h[y] >= l[y-5] or h[y] >= l[y-6] or h[y] >= l[y-7]):
                if 8 in longVal[y-16:y] or 9 in longVal[y-15:y]:
                    if c[y] < l[y-2]:
                        buyCount += 1
            if buyVal and buyVal[-1] == 13 or shortVal[y] > 8:
                buyCount = 0
            if buyCount != 0:
                if c[y] < l[y-2]:
                    buyCount += 1
                if longVal[y] == 9:
                    buyCount = 0
        buyVal.append(buyCount)
        
        if y >= 11 and sellCount == 0 and (l[y] <= h[y-3] or l[y] <= h[y-4] or l[y] <= h[y-5] or l[y] <= h[y-6] or l[y] <= h[y-7]):
            if 8 in shortVal[y-16:y] or 9 in shortVal[y-15:y]:
                if c[y] > h[y-2]:
                    sellCount = 1
        if sellVal and sellVal[-1] == 13 or longVal[y] > 8:
            sellCount = 0
        if sellCount != 0:
            if c[y] > h[y-2]:
                sellCount += 1
            if shortVal[y] == 9:
                sellCount = 0
        sellVal.append(sellCount)

    buy_deferred = apply_countdown_deferral(buyVal, h, l, c)
    sell_deferred = apply_countdown_deferral(sellVal, h, l, c)
    buy_recycled = check_recycling(buyVal, longVal)
    sell_recycled = check_recycling(sellVal, shortVal)

    agbuyVal, agsellVal, agbuyCount, agsellCount = [], [], 0, 0
    for y in range(len(c)):
        if y >= 11:
            if agbuyCount == 0 and (h[y] >= l[y-3] or h[y] >= l[y-4] or h[y] >= l[y-5] or h[y] >= l[y-6] or h[y] >= l[y-7]):
                if 8 in longVal[y-16:y] or 9 in longVal[y-15:y]:
                    if l[y] < l[y-2]:
                        agbuyCount += 1
            if agbuyVal and agbuyVal[-1] == 13 or shortVal[y] > 8:
                agbuyCount = 0
            if agbuyCount != 0:
                if l[y] < l[y-2]:
                    agbuyCount += 1
                if longVal[y] == 9:
                    agbuyCount = 0
        agbuyVal.append(agbuyCount)
        if y >= 11 and agsellCount == 0 and (l[y] <= h[y-3] or l[y] <= h[y-4] or l[y] <= h[y-5] or l[y] <= h[y-6] or l[y] <= h[y-7]):
            if 8 in shortVal[y-16:y] or 9 in shortVal[y-15:y]:
                if h[y] > h[y-2]:
                    agsellCount = 1
        if agsellVal and agsellVal[-1] == 13 or longVal[y] > 8:
            agsellCount = 0
        if agsellCount != 0:
            if h[y] > h[y-2]:
                agsellCount += 1
            if shortVal[y] == 9:
                agsellCount = 0
        agsellVal.append(agsellCount)

    # TD Buy Setup Perfection
    buy_perfection = [False] * len(buyVal)
    for i in range(len(buyVal)):
        if buyVal[i] == 9:
            if l[i] <= l[i-2] and l[i] <= l[i-3]:
                buy_perfection[i] = True
            else:
                buy_perfection[i] = False
    
    # TD Sell Setup Perfection
    sell_perfection = [False] * len(sellVal)
    for i in range(len(sellVal)):
        if sellVal[i] == 9:
            if h[i] >= h[i-2] and h[i] >= h[i-3]:
                sell_perfection[i] = True
            else:
                sell_perfection[i] = False

    # Generate Exponential Moving Average Values
    tenEMA = exponential_moving_average(c, 10)
    thirtyEMA = exponential_moving_average(c, 30)
    fiftyEMA = exponential_moving_average(c, 50)
    sixtyEMA = exponential_moving_average(c, 60)
    twohunderedEMA = exponential_moving_average(c, 200)

    # Generate Bollinger Bands
    upper_band, lower_band = bollinger_bands(c, 20, 2)

    return t, o, h, l, c, v, vol, longVal, shortVal, buyVal, sellVal, buy_deferred, sell_deferred, buy_recycled, sell_recycled, agbuyVal, agsellVal, buy_perfection, sell_perfection, \
           tenEMA, thirtyEMA, fiftyEMA, sixtyEMA, twohunderedEMA, upper_band, lower_band

@app.callback(
    [Output('graph-output', 'figure'), Output('countdown-table', 'children'), Output('aggressive-countdown-table', 'children')],
    Input('submit-button', 'n_clicks'),
    [Input('ticker-input', 'value'), Input('period-dropdown', 'value'), Input('volume-toggle', 'value'), Input('bollinger-toggle', 'value'), Input('upload-data', 'contents'), State('upload-data', 'filename')]
)
def update_graph(n_clicks, ticker, period, volume_toggle, bollinger_toggle, contents, filename):
    if contents:
        data = parse_contents(contents, filename)
    else:
        data = fetch_data(ticker, period)

    if data.empty:
        return go.Figure(), "No data available", "No data available"

    t, o, h, l, c, v, vol, longVal, shortVal, buyVal, sellVal, buy_deferred, sell_deferred, buy_recycled, sell_recycled, agbuyVal, agsellVal, buy_perfection, sell_perfection, \
        tenEMA, thirtyEMA, fiftyEMA, sixtyEMA, twohunderedEMA, upper_band, lower_band = calculate_td_sequential(data)

    fig = go.Figure(data=[go.Candlestick(x=t, open=o, high=h, low=l, close=c)])

    # Plot Exponential Moving Averages
    fig.add_trace(go.Scatter(x=t, y=fiftyEMA, mode='lines', name='50 Day EMA', line=dict(color='blue', width=1)))
    fig.add_trace(go.Scatter(x=t, y=twohunderedEMA, mode='lines', name='200 Day EMA', line=dict(color='purple', width=1)))

    # Plot bollinger bands
    if bollinger_toggle == 'on':
        fig.add_trace(go.Scatter(x=t, y=upper_band, mode='lines', name='Upper Bollinger Band', line=dict(color='magenta', width=1)))
        fig.add_trace(go.Scatter(x=t, y=lower_band, mode='lines', name='Lower Bollinger Band', line=dict(color='magenta', width=1), fill='tonexty', fillcolor='rgba(255,0,255,0.2)', showlegend=False))

    # Toggle Volume
    if volume_toggle == 'on' and any(v):
        fig.add_trace(go.Bar(x=t, y=v, name='Volume', marker_color='white', opacity=0.5))

    sell_dates = []
    buy_dates = []
    agsell_dates = []
    agbuy_dates = []
    perfection_dates = []

    for z in range(len(c)):
        if sellVal[z] == 13:
            fig.add_annotation(
                x=t[z], y=h[z]*1.07,
                text='13' + ('+' if sell_deferred[z] else '') + ('R' if sell_recycled[z] else ''),
                arrowhead=2, arrowsize=1, arrowwidth=2,
                arrowcolor='#FE642E', ax=0, ay=-50,
                font=dict(size=12, color='#FE642E'))
            sell_dates.append((t[z], c[z], sell_deferred[z], sell_recycled[z]))
        if buyVal[z] == 13:
            fig.add_annotation(
                x=t[z], y=l[z]/1.07,
                text='13' + ('+' if buy_deferred[z] else '') + ('R' if buy_recycled[z] else ''),
                arrowhead=2, arrowsize=1, arrowwidth=2,
                arrowcolor='#A9F5A9', ax=0, ay=50,
                font=dict(size=12, color='#A9F5A9'))
            buy_dates.append((t[z], c[z], buy_deferred[z], buy_recycled[z]))
        if int(shortVal[z]) == 9:
            fig.add_annotation(
                x=t[z], y=h[z]*1.04,
                text='9',
                arrowhead=2, arrowsize=1, arrowwidth=2,
                arrowcolor='#FA5858', ax=0, ay=-50,
                font=dict(size=12, color='#FA5858'))
        if int(longVal[z]) == 9:
            fig.add_annotation(
                x=t[z], y=l[z]/1.08,
                text='9',
                arrowhead=2, arrowsize=1, arrowwidth=2,
                arrowcolor='#58FA58', ax=0, ay=50,
                font=dict(size=12, color='#58FA58'))
        # Add aggressive sell and buy annotations
        if agsellVal[z] == 13:
            fig.add_annotation(
                x=t[z], y=h[z]*1.05,
                text='A13',
                arrowhead=2, arrowsize=1, arrowwidth=2,
                arrowcolor='#FF5733', ax=0, ay=-50,
                font=dict(size=12, color='#FF5733'))
            agsell_dates.append((t[z], c[z]))
        if agbuyVal[z] == 13:
            fig.add_annotation(
                x=t[z], y=l[z]/1.05,
                text='A13',
                arrowhead=2, arrowsize=1, arrowwidth=2,
                arrowcolor='#33FF57', ax=0, ay=50,
                font=dict(size=12, color='#33FF57'))
            agbuy_dates.append((t[z], c[z]))

        # Add TD Buy Setup Perfection annotations
        if buy_perfection[z] and (z == 0 or not buy_perfection[z-1]):
            fig.add_annotation(
                x=t[z], y=l[z]/1.018,  # Adjusted position
                text='P',
                showarrow=False,
                font=dict(size=12, color='#00FF00'))
            perfection_dates.append((t[z], c[z]))

        # Add TD Sell Setup Perfection annotations
        if sell_perfection[z] and (z == 0 or not sell_perfection[z-1]):
            fig.add_annotation(
                x=t[z], y=h[z]*1.018,  # Adjusted position
                text='P',
                showarrow=False,
                font=dict(size=12, color='#FF0000'))
            perfection_dates.append((t[z], c[z]))

    # Annotate the latest price
    latest_price = c[-1]
    if latest_price < c[-2]:
        fig.add_annotation(
            x=t[-1], y=latest_price,
            text=f'{latest_price:.2f}',
            showarrow=False,
            font=dict(size=10, color='#FE2E2E'))
    else:
        fig.add_annotation(
            x=t[-1], y=latest_price,
            text=f'{latest_price:.2f}',
            showarrow=False,
            font=dict(size=10, color='#088A29'))

    # Display the highest and lowest prices for the time interval specified
    idx_min = int(np.argmin(l))
    idx_max = int(np.argmax(h))
    fig.add_annotation(
        x=t[idx_min], y=h[idx_max]*0.9,
        text=f"L: ${l[idx_min]:.2f}",
        showarrow=False,
        font=dict(size=10, color='#FE2E2E'))
    fig.add_annotation(
        x=t[idx_max], y=h[idx_max]*0.98,
        text=f"H: ${h[idx_max]:.2f}",
        showarrow=False,
        font=dict(size=10, color='#2EFE2E'))

    # Remove the ticker name from the plot title when data is uploaded
    title = 'Candlestick Chart' if contents else f'{ticker} Candlestick Chart'

    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_dark',
        xaxis_rangeslider_visible=False
    )

    # Create table rows for normal countdowns
    normal_trades = sorted(sell_dates + buy_dates, key=lambda x: x[0])
    normal_table_rows = []
    for date, price, deferred, recycled in normal_trades:
        recycle_text = 'R' if recycled else ''
        row_style = {'backgroundColor': '#FFDDDD' if (date, price, deferred, recycled) in sell_dates else '#DDFFDD'}
        if (date, price, deferred, recycled) in sell_dates:
            normal_table_rows.append(html.Tr([html.Td("Sell Countdown"), html.Td(date.strftime('%Y-%m-%d')), html.Td(f'{price:.2f}'), html.Td('+' if deferred else ''), html.Td(recycle_text)], style=row_style))
        if (date, price, deferred, recycled) in buy_dates:
            normal_table_rows.append(html.Tr([html.Td("Buy Countdown"), html.Td(date.strftime('%Y-%m-%d')), html.Td(f'{price:.2f}'), html.Td('+' if deferred else ''), html.Td(recycle_text)], style=row_style))

    normal_table = html.Table([
        html.Thead(html.Tr([html.Th("Type"), html.Th("Date"), html.Th("Price"), html.Th("Deferred"), html.Th("Recycled")])),
        html.Tbody(normal_table_rows)
    ])

    # Create table rows for aggressive countdowns
    aggressive_trades = sorted(agsell_dates + agbuy_dates, key=lambda x: x[0])
    aggressive_table_rows = []
    for date, price in aggressive_trades:
        row_style = {'backgroundColor': '#FFDDDD' if (date, price) in agsell_dates else '#DDFFDD'}
        if (date, price) in agsell_dates:
            aggressive_table_rows.append(html.Tr([html.Td("Aggressive Sell Countdown"), html.Td(date.strftime('%Y-%m-%d')), html.Td(f'{price:.2f}')], style=row_style))
        if (date, price) in agbuy_dates:
            aggressive_table_rows.append(html.Tr([html.Td("Aggressive Buy Countdown"), html.Td(date.strftime('%Y-%m-%d')), html.Td(f'{price:.2f}')], style=row_style))

    aggressive_table = html.Table([
        html.Thead(html.Tr([html.Th("Type"), html.Th("Date"), html.Th("Price")])),
        html.Tbody(aggressive_table_rows)
    ])

    return fig, normal_table, aggressive_table

if __name__ == '__main__':
    app.run_server(debug=True, port=8057)
    
