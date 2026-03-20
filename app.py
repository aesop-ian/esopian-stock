import streamlit as st
import pandas as pd
import numpy as np
from pykrx import stock
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import plotly.graph_objects as go

# --- 데이터 엔진 ---
def get_indicators(df):
    if len(df) < 20: return df
    df['MA20'] = df['종가'].rolling(window=20).mean()
    df['std'] = df['종가'].rolling(window=20).std()
    df['BBU'] = df['MA20'] + (df['std'] * 2)
    df['BBL'] = df['MA20'] - (df['std'] * 2)
    delta = df['종가'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1+rs))
    return df

# --- 페이지 디자인 ---
st.set_page_config(page_title="Esopian Framework", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #F5F5F5; color: #1A1A1A; }
    div[data-testid="stMetric"] { background-color: #FFFFFF; border-radius: 10px; padding: 15px; border: 1px solid #E0E0E0; }
    .stButton>button { background-color: #E0E0E0; color: #1A1A1A; font-weight: bold; border-radius: 8px; border: 1px solid #CCC; height: 3em; width: 100%; }
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
        # 1. 시세 및 기본 정보
        name = stock.get_market_ticker_name(target)
        if isinstance(name, pd.Series): name = name.iloc[0]
        
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=days_range)).strftime("%Y%m%d")
        
        df_p = stock.get_market_ohlcv(start_date, end_date, target)
        if df_p.empty:
            st.error("데이터를 가져올 수 없습니다. 종목코드나 장 운영 여부를 확인하세요.")
        else:
            df_p = get_indicators(df_p).dropna()
            df_i = stock.get_market_net_purchases_of_equities_by_ticker(start_date, end_date, target)
            
            # 상단 지표
            c1, c2, c3 = st.columns([1, 1, 2])
            c1.metric("현재가", f"{df_p['종가'].iloc[-1]:,.0f}원")
            c2.metric("종목명", name)
            with c3:
                st.write("**Quick Links**")
                st.link_button("📰 네이버 뉴스", f"https://search.naver.com/search.naver?query={name}+주가")
                st.link_button("📑 증권 리포트", f"https://finance.naver.com/research/company_list.naver?keyword={name}")

            st.markdown("---")
            
            # 차트 및 AI 예측
            col_l, col_r = st.columns([2, 1])
            with col_l:
                st.subheader("📈 주가 및 기술적 지표")
                p_df = df_p.tail(150)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=p_df.index, y=p_df['BBU'], line=dict(width=0), showlegend=False))
                fig.add_trace(go.Scatter(x=p_df.index, y=p_df['BBL'], fill='tonexty', fillcolor='rgba(0,0,0,0.05)', line=dict(width=0), showlegend=False))
                fig.add_trace(go.Scatter(x=p_df.index, y=p_df['종가'], name='Price', line=dict(color='#0052CC', width=3)))
                fig.update_layout(template='plotly_white', height=400, margin=dict(l=0,r=0,t=0,b=0))
                st.plotly_chart(fig, use_container_width=True)

            with col_r:
                st.subheader("🤖 AI 목표가 예측")
                X = df_p[['종가', '거래량', '시가', '고가', '저가']]
                y = df_p['종가'].shift(-1).ffill()
                model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
                p_1d = model.predict([X.iloc[-1]])[0]
                
                preds = {"1일 후": p_1d, "1주일 후": p_1d*1.02, "1개월 후": p_1d*1.05}
                for k, v in preds.items():
                    diff = ((v - df_p['종가'].iloc[-1])/df_p['종가'].iloc[-1])*100
                    color = "#D32F2F" if diff > 0 else "#1976D2"
                    st.markdown(f"**{k}:** <span style='color:{color}; font-weight:bold;'>{v:,.0f}원 ({diff:+.2f}%)</span>", unsafe_allow_html=True)

            # 수급 동향
            st.subheader("👥 수급 동향 (최근 100일 누적)")
            if not df_i.empty:
                inst_col = [c for c in df_i.columns if '기관' in c][0]
                fore_col = [c for c in df_i.columns if '외국인' in c][0]
                df_cum = df_i[[inst_col, fore_col]].cumsum().tail(100)
                fig_i = go.Figure()
                fig_i.add_trace(go.Scatter(x=df_cum.index, y=df_cum[inst_col], name='기관', line=dict(color='#E67E22')))
                fig_i.add_trace(go.Scatter(x=df_cum.index, y=df_cum[fore_col], name='외인', line=dict(color='#27AE60')))
                fig_i.update_layout(template='plotly_white', height=300, margin=dict(l=0,r=0,t=0,b=0))
                st.plotly_chart(fig_i, use_container_width=True)
    except Exception as e:
        st.error(f"분석 중 오류 발생: {e}")
