import streamlit as st
#st.set_page_config(page_title="UniSwapV3 Impermanent Loss", layout="wide")
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import numpy as np


st.title("Uniswap V3 Impermanent Loss Hedging")
st.subheader("Strategy: Enter LP at max price at 100% USD, hedging with put option at DCA strike")

P = st.number_input('Max Price', value=1200)
P_max = P

col1, col2 = st.columns(2)
P_min = col1.number_input("Min Price",value=800)

y = st.number_input('USD Token in Pool', value=100)
L=(np.sqrt(P_min)/P + 1/np.sqrt(P))/(1-P_min/P)*y
#x_min = 100% position in token x when price hits P_min                                                                                 
x_min = L*(1/np.sqrt(P_min)-1/np.sqrt(P_max))
P_DCA = y/x_min

lower_bound = P_min/10000
upper_bound = 2.1*P_max
step = upper_bound/10000
prices = np.arange(P_min/1000,2*P_max+1,step)

pnl = []
for price in prices:
    x_future = max(0,L*(min(1/np.sqrt(P_min),1/np.sqrt(price))-1/np.sqrt(P_max)))
    y_future = max(0,L*(min(np.sqrt(P_max),np.sqrt(price))-np.sqrt(P_min)))
    pnl.append(y_future + x_future * price - y)

df=pd.DataFrame()
df["Price"] = prices
df["PNL"] = pnl

fig = px.line(df, x="Price", y="PNL", title='PNL without hedging')
fig.update_xaxes(title_text='Price')
fig.update_yaxes(title_text='PNL')
st.plotly_chart(fig,use_container_width=True)


pnl_hedged = []
for price in prices:
    x_future = max(0,L*(min(1/np.sqrt(P_min),1/np.sqrt(price))-1/np.sqrt(P_max)))
    y_future = max(0,L*(min(np.sqrt(P_max),np.sqrt(price))-np.sqrt(P_min)))
    pnl_hedged.append(y_future + x_future * price - y + x_min * max(0,P_DCA-price))

st.subheader("Hedging left tail of LP position with put option, strike = DCA price")

col1, col2 = st.columns(2)
col1.write("Strike (DCA price when price of token hits min price)")
col2.text(P_DCA)
col1, col2 = st.columns(2)
col1.write("Number of puts:")
col2.text(x_min)

df = pd.DataFrame()
df["Price"] = prices
df["PNL"] = pnl_hedged

#df = px.data.gapminder().query("country=='Canada'")
fig = px.line(df, x="Price", y="PNL", title='PNL, LP with put hedging')
fig.update_xaxes(title_text='Price')
fig.update_yaxes(title_text='PNL')
st.plotly_chart(fig,use_container_width=True)