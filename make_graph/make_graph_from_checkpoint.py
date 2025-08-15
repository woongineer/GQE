import torch
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio

checkpoint_path = "fix_sample_SM_temp_schedule_checkpoint_8000.pt"
output_name = "main_temp_schedule"
window = 50

ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

fidelity_values = ckpt["fidelity_history"]

df = pd.DataFrame({'Fidelity': fidelity_values})
df['Moving Average'] = df['Fidelity'].rolling(window=window, min_periods=1).mean()

fig = go.Figure()
fig.add_trace(go.Scatter(
    y=df['Fidelity'],
    mode='lines',
    name='Fidelity',
    line=dict(color='yellow')
))
fig.add_trace(go.Scatter(
    y=df['Moving Average'],
    mode='lines',
    name=f'Moving Average (window={window})',
    line=dict(color='red')
))
fig.update_layout(
    title='Fidelity and Moving Average',
    xaxis_title='Iteration',
    yaxis_title='Fidelity',
    template='plotly_white'
)

# HTML 저장
html_output_path = f"{output_name}_fid_plot.html"
pio.write_html(fig, file=html_output_path, auto_open=False)

print(f"Graph saved to {html_output_path}")
