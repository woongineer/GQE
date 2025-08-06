import re
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio

# 로그 파일 경로
log_path = "main.log"  # 필요에 따라 절대경로로 수정하세요

# 로그 파일 읽기
with open(log_path, "r") as file:
    log_data = file.read()

# Ave True E 값 정규표현식으로 추출
true_e_values = [
    float(match.group(1))
    for match in re.finditer(r"Ave True E: ([\d\.]+)", log_data)
]

# DataFrame 생성 및 이동평균 계산
df = pd.DataFrame({'Ave True E': true_e_values})
df['Moving Average'] = df['Ave True E'].rolling(window=50).mean()

# Plotly 그래프 생성
fig = go.Figure()

# 노란색: raw Ave True E
fig.add_trace(go.Scatter(
    y=df['Ave True E'],
    mode='lines',
    name='Ave True E',
    line=dict(color='yellow')
))

# 빨간색: 이동평균
fig.add_trace(go.Scatter(
    y=df['Moving Average'],
    mode='lines',
    name='Moving Average (window=50)',
    line=dict(color='red')
))

# 레이아웃 설정
fig.update_layout(
    title='Ave True E and Moving Average',
    xaxis_title='Iteration',
    yaxis_title='Ave True E',
    template='plotly_white'
)

# HTML로 저장
html_output_path = "ave_true_e_plot.html"
pio.write_html(fig, file=html_output_path, auto_open=False)

print(f"그래프가 저장되었습니다: {html_output_path}")
