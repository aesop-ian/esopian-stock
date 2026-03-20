import os
import sys
import subprocess
import time

# [1] 시스템 부품(setuptools) 강제 주입 및 경로 갱신
def force_install_setuptools():
    try:
        import pkg_resources
    except ImportError:
        # 패키지가 없으면 강제 설치
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "setuptools==69.5.1"])
        # 설치 후 시스템이 바로 인식하도록 경로 강제 업데이트
        import site
        from importlib import reload
        reload(site)
        # 다시 시도
        try:
            import pkg_resources
        except ImportError:
            st.error("시스템 부품 설치에 실패했습니다. 관리자에게 문의하세요.")

import streamlit as st

# 실행하자마자 부품 체크
force_install_setuptools()

# [2] 이제 나머지 라이브러리 로드
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

# pykrx는 여기서 지연 임포트 (에러 방지 핵심)
try:
    from pykrx import stock
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pykrx"])
    from pykrx import stock

from sklearn.ensemble import RandomForestRegressor

# --- 디자인 및 설정 ---
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

# --- 사이드바 ---
st.sidebar.header("📊 분석 설정")
target = st.sidebar.text_input("종목코드", value="141080")
days_range = st.sidebar.slider("AI 학습 기간 (일)", 200, 1000, 500)
analyze_btn = st.sidebar.button("전략 분석 실행")

if analyze_btn:
    try:
        with st.spinner('Framework 엔진 가동 중...'):
            name = stock.get_market_ticker_name(target)
            if isinstance(name, pd.Series): name = name.iloc[0]
            
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=days_range)).strftime("%Y%m%d")
            
            df_p = stock.get_market_ohlcv(start_date, end_date, target)
            
            c1, c2 = st.columns(2)
            c1.metric("현재가", f"{df_p['종가'].iloc[-1]:,.0f}원")
            c2.metric("종목명", name)

            st.markdown("---")
            st.subheader("📈 주가 추세")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_p.index, y=df_p['종가'], name='Price', line=dict(color='#0052CC', width=3)))
            fig.update_layout(template='plotly_white', height=400)
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"분석 중 오류 발생: {e}")
