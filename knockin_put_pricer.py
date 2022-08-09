import streamlit as st
st.set_page_config(page_title="Black Scholes Pricer, Knockin Put Option", layout="wide")
import numpy as np
import pandas as pd
import scipy.stats as si
import plotly.express as px
import plotly.graph_objects as go

st.title("Black Scholes Pricer, Knockin Put Option")

P = st.number_input('Current Price', value=1666.000, step = 0.001)
K_put = st.number_input('Put Strike', value=1470.000, step = 0.001)
B_knockin = st.number_input('Down and In Barrier', value=1333.000, step = 0.001)
numberdays = st.number_input('Number of days', value=7.00, step = .01)
T = numberdays/365.25

#enter drift (risk-free rate for option pricing)
r_in_percent = st.number_input('Drift in % (risk-free rate for option pricing)', value=0.0000, step = 0.0001)
r = r_in_percent/100
sigma_B_in_percent = st.number_input('Implied Volatility at down and in barrier in %', value = 90.000, step=0.0001)
sigma_B = sigma_B_in_percent / 100
sigma_K_in_percent = st.number_input('Implied Volatility at put strike, in %', value = 90.000, step=0.0001)
sigma_K = sigma_K_in_percent / 100

def euro_option(S, K, T, r, sigma, option = 'call'):
    
    #S: spot price
    #K: strike price
    #T: time to maturity
    #r: interest rate
    #sigma: volatility of underlying asset
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    if option == 'call':
        result = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    if option == 'put':
        result = (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0))
    if option == 'binary put':
        result = (np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0))        
        
    return result

def knockin_put(S, K, T, r, sigma, B):
  return (K - B) * euro_option(S, B, T, r, sigma, 'binary put') + euro_option(P, B, T, r, sigma, 'put')

lower_bound = 0
upper_bound = 2*K_put
step = upper_bound/10000
prices = np.arange(lower_bound,upper_bound+1,step)

put_vanilla_barrier_strike = []
put_binary = []
put_knockin = []
put_vanilla = []
for price in prices:
    put_vanilla_barrier_strike.append(max(0,B_knockin-price))
    put_binary.append(K_put - B_knockin if price <= B_knockin else 0)
    put_knockin.append(K_put-price if price <= B_knockin else 0)
    put_vanilla.append(max(0,K_put-price))    

df = pd.DataFrame()
df["Price"] = prices
df["Vanilla Put Payoff, Barrier Strike"] = put_vanilla_barrier_strike
df["Binary Put Payoff"] = put_binary
df["Knockin Put Payoff"] = put_knockin
df["Vanilla Put Payoff"] = put_vanilla

col1, col2 = st.columns(2)
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["Price"], y=df["Vanilla Put Payoff, Barrier Strike"], mode='lines', name="Vanilla Put Payoff"))
fig.add_trace(go.Scatter(x=df["Price"], y=df["Binary Put Payoff"], mode='lines', name="Binary Put Payoff"))
fig.add_trace(go.Scatter(x=df["Price"], y=df["Knockin Put Payoff"], mode='lines', name="Knockin Put Payoff"))


fig.update_layout(title=dict(text='Options payoff: Decomposition of Knockin put option = Vanilla put option + binary put option<br>Knock in put: Strike = ' +str(K_put) + ", Barrier = " +str(B_knockin) + '<br>Vanilla put: Strike = ' +str(B_knockin)+'<br>Binary put: Strike = ' +str(B_knockin) + ', Notional = ' +str(K_put - B_knockin),
font=dict(size=12)))
fig.update_xaxes(title_text='Price')
fig.update_yaxes(title_text='Payoff')
col1.plotly_chart(fig,use_container_width=True)

fig = px.line(df, x="Price", y="Vanilla Put Payoff", title='Vanilla Put option, Strike = ' +str(K_put))
fig.update_xaxes(title_text='Price')
fig.update_yaxes(title_text='Payoff')
col2.plotly_chart(fig,use_container_width=True)

df = pd.DataFrame([["Vanilla put, Strike = " + str(B_knockin), euro_option(P, B_knockin, T, r, sigma_B, 'put'),euro_option(P, B_knockin, T, r, sigma_B, 'put')/K_put],
                   ["Binary put, Strike = "+ str(B_knockin) +", Notional = " +str(K_put - B_knockin), (K_put - B_knockin) * euro_option(P, B_knockin, T, r, sigma_B, 'binary put'),(K_put - B_knockin) * euro_option(P, B_knockin, T, r, sigma_B, 'binary put')/K_put],
                  ["Knockin put, Strike = "+ str(K_put) +", lower barrier = " +str(B_knockin) + " (Sum of above two options)", knockin_put(P, K_put, T, r, sigma_B, B_knockin), knockin_put(P, K_put, T, r, sigma_B, B_knockin)/K_put],
                  ["Compare with: Vanilla put, Strike = " + str(K_put), euro_option(P, K_put, T, r, sigma_K, 'put'), euro_option(P, K_put, T, r, sigma_K, 'put')/K_put]],
                  columns=["Hedging Instrument", "Black-Scholes Price", "Percentage of notional, notional = " +str(K_put) + " (strike put)"])

df["Percentage of notional, notional = " +str(K_put) + " (strike put)"] = df["Percentage of notional, notional = " +str(K_put) + " (strike put)"].map(lambda n: '{:,.3%}'.format(n))
st.dataframe(df)

st.write("Discount: " ,   "{:.02%}".format(1-knockin_put(P, K_put, T, r, sigma_B, B_knockin)/euro_option(P, K_put, T, r, sigma_K, 'put')) )