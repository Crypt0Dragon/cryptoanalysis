import streamlit as st
#st.set_page_config(page_title="UniSwapV3 Impermanent Loss", layout="wide")
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import numpy as np
import math

st.title("Uniswap V3 Impermanent Loss")
st.subheader("Input")

P = st.number_input('Current Price', value=1200)
col1, col2 = st.columns(2)
P_min = col1.number_input("Min Price",value=1000)
P_max = col2.number_input("Max Price",value=1200)

y = st.number_input('Token1 in Pool', value=100)
L=(math.sqrt(P_min)/P + 1/math.sqrt(P))/(1-P_min/P)*y                                                                                 
x=max(0,(math.sqrt(P_min)/P - 1/math.sqrt(P_max))*L+y/P)
st.write("Token0 in Pool:")
st.text(x)

prices = np.arange(1,2*P_max+1,1)
loss = []
for price in prices:
    x_future = max(0,L*(min(1/math.sqrt(P_min),1/math.sqrt(price))-1/math.sqrt(P_max)))
    y_future = max(0,L*(min(math.sqrt(P_max),math.sqrt(price))-math.sqrt(P_min)))
    loss.append(-y_future - x_future * price + y+x*price)

df=pd.DataFrame()
df["Price"] = prices
df["Impermanent Loss"] = loss

fig = px.line(df, x="Price", y="Impermanent Loss", title='Impermanent Loss Chart (Token1 denominated)')
fig.update_xaxes(title_text='Price')
fig.update_yaxes(title_text='Impermanent Loss')
st.plotly_chart(fig,use_container_width=True)

st.subheader("Impermanent Loss Calculation")

P_future= st.number_input('Future Price', value=P)

x_future = max(0,L*(min(1/math.sqrt(P_min),1/math.sqrt(P_future))-1/math.sqrt(P_max)))
y_future = max(0,L*(min(math.sqrt(P_max),math.sqrt(P_future))-math.sqrt(P_min)))



st.subheader("If In Pool")
col1, col2 = st.columns(2)
col1.write("Token0")
col2.text(x_future)
col1, col2 = st.columns(2)
col1.write("Token1")
col2.text(y_future)
col1, col2 = st.columns(2)
col1.write("Token0 Value")
col2.text(x_future * P_future)
col1, col2 = st.columns(2)
col1.write("All Value")
col2.text(y_future + x_future * P_future)

st.subheader("If Not In Pool")
col1, col2 = st.columns(2)
col1.write("Token0 Value")
col2.text(x*P_future)
col1, col2 = st.columns(2)
col1.write("All Value")
col2.text(y+x*P_future)
 
col1, col2 = st.columns(2)
col1.subheader("Loss")
col2.subheader(-y_future - x_future * P_future + y+x*P_future)