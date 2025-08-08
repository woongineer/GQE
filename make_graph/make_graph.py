import re
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio

name = "main"
log_path = f"{name}.log"

with open(log_path, "r") as file:
    log_data = file.read()

true_e_values = [
    float(match.group(1))
    for match in re.finditer(r"Ave True E: ([\d\.]+)", log_data)
]

df = pd.DataFrame({'Ave True E': true_e_values})
df['Moving Average'] = df['Ave True E'].rolling(window=50).mean()

fig = go.Figure()

fig.add_trace(go.Scatter(
    y=df['Ave True E'],
    mode='lines',
    name='Ave True E',
    line=dict(color='yellow')
))

fig.add_trace(go.Scatter(
    y=df['Moving Average'],
    mode='lines',
    name='Moving Average (window=50)',
    line=dict(color='red')
))

fig.update_layout(
    title='Ave True E and Moving Average',
    xaxis_title='Iteration',
    yaxis_title='Ave True E',
    template='plotly_white'
)

html_output_path = f"{name}_ave_true_e_plot.html"
pio.write_html(fig, file=html_output_path, auto_open=False)

print(f"graph saved")
