import os
import subprocess
import sys

# [1] 시스템 패키지 강제 복구 로직 (가장 중요)
try:
    import pkg_resources
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "setuptools", "wheel"])
    import pkg_resources

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

# [2] pykrx 임포트 (오류 방지를 위해 지연 임포트)
try:
    from pykrx import stock
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pykrx"])
    from pykrx import stock

from sklearn.ensemble import RandomForestRegressor

# --- 데이터 엔진 ---
def get_indicators(df):
    if len(df) < 20: return df
    df['MA20'] = df['종가'].rolling(window=20).mean()
    df['std'] = df['종가'].rolling(window=20).std()
    df['BBU'] = df['MA20'] + (df['std'] * 2)
    df['BBL'] = df['MA20'] - (df['std'] * 2)
    return df

# --- 페이지 설정 및 디자인 ---
st.set_page_config(page_title="Esopian Framework", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #F5F5F5; color: #1A1A1A; }
    div[data-testid="stMetric"] { background-color: #FFFFFF; border-radius: 10px; padding: 15px; border: 1px solid #E0E0E0; }
    .stButton>button { background-color: #E0E0E0 !important; color: #1A1A1A !important; font-weight: bold; border-radius: 8px; border: 1px solid #CCC; height: 3em; width: 100%; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ Esopian Stock Analysis Framework")
st.caption("AI Forecasting & Quantitative Analysis Terminal")

# --- 사이드바 및 실행 ---
st.sidebar.header("📊 분석 설정")
target = st.sidebar.text_input("종목코드", value="141080")
days_range = st.sidebar.slider("AI 학습 기간 (일)", 200, 1000, 500)
analyze_btn = st.sidebar.button("전략 분석 실행")

if analyze_btn:
    try:
        with st.spinner('데이터를 분석 중입니다...'):
            name = stock.get_market_ticker_name(target)
            if isinstance(name, pd.Series): name = name.iloc[0]
            
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=days_range)).strftime("%Y%m%d")
            
            df_p = stock.get_market_ohlcv(start_date, end_date, target)
            df_p = get_indicators(df_p).dropna()
            df_i = stock.get_market_net_purchases_of_equities_by_ticker(start_date, end_date, target)
            
            c1, c2, c3 = st.columns([1, 1, 2])
            c1.metric("현재가", f"{df_p['종가'].iloc[-1]:,.0f}원")
            c2.metric("종목명", name)
            with c3:
                st.write("**Quick Links**")
                st.link_button("📰 네이버 뉴스", f"https://search.naver.com/search.naver?query={name}+주가")
                st.link_button("📑 증권 리포트", f"https://finance.naver.com/research/company_list.naver?keyword={name}")

            st.markdown("---")
            
            col_l, col_r = st.columns([2, 1])
            with col_l:
                st.subheader("📈 주가 추세")
                p_df = df_p.tail(150)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=p_df.index, y=p_df['종가'], name='Price', line=dict(color='#0052CC', width=3)))
                fig.update_layout(template='plotly_white', height=400, margin=dict(l=0,r=0,t=0,b=0))
                st.plotly_chart(fig, use_container_width=True)

            with col_r:
                st.subheader("🤖 AI 목표가")
                X = df_p[['종가', '거래량', '시가', '고가', '저가']]
                y = df_p['종가'].shift(-1).ffill()
                model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
                p_1d = model.predict([X.iloc[-1]])[0]
                st.info(f"내일 예상가: {p_1d:,.0f}원")
    except Exception as e:
        st.error(f"오류 발생: {e}")
