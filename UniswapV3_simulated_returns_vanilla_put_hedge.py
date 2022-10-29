import streamlit as st
st.set_page_config(page_title="UniSwapV3 Simulated Returns", layout="wide")
import numpy as np
import plotly.express as px
import pandas as pd
import scipy.stats as si

st.title("UniSwapV3 Simulated Returns, LP hedging with vanilla put option, enter LP at max price at 100% USD")

P = st.number_input('Current Price', value=1666.000, step = 0.001)
y = st.number_input('Number of USD tokens', value=1000000, step = 1)

col1, col2 = st.columns(2)
P_min = col1.number_input("Min Price",value=1333.000, step = 0.001)
P_max = col2.number_input("Max Price",value=1650.000, step = 0.001)

if P_max > P:
    st.write("Max Price has to be less or equal Current Price") 

else: 

    L=(np.sqrt(P_min)/min(P,P_max) + 1/np.sqrt(min(P,P_max)))/(1-P_min/min(P,P_max))*y                                                                                 
    x_min = L*(1/np.sqrt(P_min)-1/np.sqrt(P_max))
    P_DCA = y/x_min
    col1, col2 = st.columns(2)
    col1.write("Strike (DCA price when price of token hits min price)")
    col2.text(P_DCA)
    col1, col2 = st.columns(2)
    col1.write("Number of puts:")
    col2.text(x_min)

    st.subheader("Enter your hedge: ")

    col1, col2 = st.columns(2)
    #Set your own number of puts
    x_min_new = col1.number_input('Number of puts', value=x_min, step = 0.001)

    #Set your own put strike 
    P_put_new = col2.number_input('Put strike', value=P_DCA, step = 0.001)

    st.subheader("Enter your Monte-Carlo simulation parameters: ")

    r_in_percent = st.number_input('Drift in % (risk-free rate for option pricing)', value=0.0000, step = 0.0001)
    r = r_in_percent/100
    sigma_in_percent = st.number_input('Volatility in %', value = 90.000, step=0.0001)
    sigma = sigma_in_percent / 100
    numberdays = st.number_input("Number of days", value = 7, step = 1)
    steps_per_day = st.number_input("Number of time steps per days for simulation", value = 20, step = 1)
    number_paths = st.number_input("Number of simulated paths", value = 10000, step = 1)
    APR_in_percent  = st.number_input("Enter APR in %", value = 250.00, step = 0.001)
    APR = APR_in_percent/100



    lower_bound = P_min/10000
    upper_bound = 2.1*P_max
    step = upper_bound/10000
    prices = np.arange(P_min/1000,2*P_max+1,step)

    pnl = []
    for price in prices:
        x_future = max(0,L*(min(1/np.sqrt(P_min),1/np.sqrt(price))-1/np.sqrt(P_max)))
        y_future = max(0,L*(min(np.sqrt(P_max),np.sqrt(price))-np.sqrt(P_min)))
        pnl.append(y_future + x_future * price - y)

    df = pd.DataFrame()
    df["Price"] = prices
    df["PNL"] = pnl

    col1, col2 =st.columns(2)
    fig = px.line(df, x="Price", y="PNL", title='Impermanent loss without hedging')
    fig.update_xaxes(title_text='Price')
    fig.update_yaxes(title_text='PNL')
    st.plotly_chart(fig,use_container_width=True)

    pnl_hedged = []
    for price in prices:
        x_future = max(0,L*(min(1/np.sqrt(P_min),1/np.sqrt(price))-1/np.sqrt(P_max)))
        y_future = max(0,L*(min(np.sqrt(P_max),np.sqrt(price))-np.sqrt(P_min)))
        pnl_hedged.append(y_future + x_future * price  - y + x_min_new * max(0,P_put_new-price))

    df = pd.DataFrame()
    df["Price"] = prices
    df["PNL"] = pnl_hedged

    fig = px.line(df, x="Price", y="PNL", title='Impermanent loss, LP with put hedging<br>Number of puts: ' + str(x_min_new) + ', Strike = ' + str(P_put_new))
    fig.update_xaxes(title_text='Price')
    fig.update_yaxes(title_text='PNL')
    st.plotly_chart(fig,use_container_width=True)


    with st.form("generate_mc_simulaton"):

        submitted = st.form_submit_button("Generate Monte Carlo Simulation")
        if submitted:
            st.subheader("Simulated results")
            def gen_paths(S0, r, sigma, T, M, I):
                dt = float(T) / M
                paths = np.zeros((M + 1, I), np.float64)
                paths[0] = S0
                for t in range(1, M + 1):
                    rand = np.random.standard_normal(I)
                    paths[t] = paths[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt +
                                                    sigma * np.sqrt(dt) * rand)
                return paths

            def portfolio_value(price):
                x_future = max(0,L*(min(1/np.sqrt(P_min),1/np.sqrt(price))-1/np.sqrt(P_max)))
                y_future = max(0,L*(min(np.sqrt(P_max),np.sqrt(price))-np.sqrt(P_min)))
                return y_future + x_future * price  

            #Starting price
            S0 = P

            # time to maturity expressed in years
            T = numberdays/365.25

            # total number of time steps
            M = numberdays * steps_per_day

            # time interval for each time step, expressed in years
            dt = T / M

            mc_paths = pd.DataFrame(gen_paths(S0,r,sigma,T,M,number_paths).transpose())
            mc_portfolio_values = pd.DataFrame()
            for i in range(0, M + 1):
                mc_portfolio_values_time = []
                prices = mc_paths[i]
                for j in range(0, number_paths):
                    mc_portfolio_values_time.append(portfolio_value(prices[j]))
                mc_portfolio_values = pd.concat([mc_portfolio_values, pd.Series(mc_portfolio_values_time)], axis = 1)
            mc_portfolio_values.columns = range(0 , M + 1)    

            earnings = pd.DataFrame()
            for i in range(0, M + 1):
                earnings_time = []
                portfolio_values = mc_portfolio_values[i]
                for j in range(0, number_paths):
                    if mc_paths[i][j] < P_min or mc_paths[i][j] > P_max:
                        earnings_time.append(0)
                    else:
                        earnings_time.append(APR*dt*portfolio_values[j])
                earnings = pd.concat([earnings, pd.Series(earnings_time)], axis = 1)
            earnings.columns = range(0, M+1)    

            #rows: Monte carlo paths
            #columns: generated fee at time step n = value of LP position* APR * dt if price in range
            earnings["Total earnings"] = earnings.sum(axis=1) 
            earnings["Number of days in range"] = earnings.gt(0).sum(axis=1) * 1/steps_per_day

            col1, col2, col3 = st.columns([2,1,1])        
            df = px.data.tips()
            fig = px.histogram(earnings["Total earnings"], title = "Fees generated<br>Number of simulations: " + str(number_paths))
            col1.plotly_chart(fig,use_container_width=True)

            col2.text("Average fees generated: ")
            col3.text(earnings["Total earnings"].mean())
            col2.text("Median fees generated: ")
            col3.text(earnings["Total earnings"].median())
            col2.text("5th percentile fees generated: ")
            col3.text(earnings["Total earnings"].quantile(.05))
            col2.text("1st percentile fees generated: ")
            col3.text(earnings["Total earnings"].quantile(.01))

            col1, col2, col3 = st.columns([2,1,1])      
            df = px.data.tips()
            fig = px.histogram(earnings["Number of days in range"], title = "Number of days in range<br>Number of simulations: " + str(number_paths))
            col1.plotly_chart(fig,use_container_width=True)


            col2.text("Average number of days in range: ")
            col3.text(earnings["Number of days in range"].mean())

            col2.text("Median number of days in range: ")
            col3.text(earnings["Number of days in range"].median())

            col2.text("5th percentile number of days in range: ")
            col3.text(earnings["Number of days in range"].quantile(.05))

            col2.text("1st percentile number of days in range: ")
            col3.text(earnings["Number of days in range"].quantile(.01))

            mc_paths["earnings"] = earnings["Total earnings"]
            pnl = []
            for price in mc_paths[M]:
                x_future = max(0,L*(min(1/np.sqrt(P_min),1/np.sqrt(price))-1/np.sqrt(P_max)))
                y_future = max(0,L*(min(np.sqrt(P_max),np.sqrt(price))-np.sqrt(P_min)))
                pnl.append(y_future + x_future * price - y)
            mc_paths["PnL"] = pnl + mc_paths["earnings"]    

            col1, col2, col3 = st.columns([2,1,1])
            df = px.data.tips()
            fig = px.histogram(mc_paths["PnL"], title = "PnL including generated fees, unhedged<br>Number of simulations: " + str(number_paths))
            col1.plotly_chart(fig,use_container_width=True)
            
            col2.text("Average PnL, unhedged LP: ")
            col3.text(mc_paths["PnL"].mean())

            col2.text("Median PnL, unhedged LP: ")
            col3.text(mc_paths["PnL"].median())

            col2.text("5th percentile PnL, unhedged LP: ")
            col3.text(mc_paths["PnL"].quantile(.05))

            col2.text("1st percentile PnL, unhedged LP: ")
            col3.text(mc_paths["PnL"].quantile(.01))

            pnl_hedged = []
            for price in mc_paths[M]:
                x_future = max(0,L*(min(1/np.sqrt(P_min),1/np.sqrt(price))-1/np.sqrt(P_max)))
                y_future = max(0,L*(min(np.sqrt(P_max),np.sqrt(price))-np.sqrt(P_min)))
                pnl_hedged.append(y_future + x_future * price  - y + x_min_new * max(0,P_put_new-price))
            mc_paths["PnL hedged"] = pnl_hedged + mc_paths["earnings"]    

            col1, col2, col3 = st.columns([2,1,1])
            df = px.data.tips()
            fig = px.histogram(mc_paths["PnL hedged"], title = "PnL including generated fees, hedged with " + str(x_min_new) + " puts, Strike: " +str(P_put_new) + "<br>Number of simulations: " + str(number_paths))
            col1.plotly_chart(fig,use_container_width=True)

            col2.text("Average PnL, hedged LP: ")
            col3.text(mc_paths["PnL hedged"].mean())

            col2.text("Median PnL, hedged LP: ")
            col3.text(mc_paths["PnL hedged"].median())

            col2.text("5th percentile PnL, hedged LP: ")
            col3.text(mc_paths["PnL hedged"].quantile(.05))

            col2.text("1st percentile PnL, hedged LP: ")
            col3.text(mc_paths["PnL hedged"].quantile(.01))

            mc_paths["Price End"] = mc_paths[M]
            values_put = []

            for i in range(0, len(mc_paths)):
                values_put.append(max(0,P_put_new-float(mc_paths["Price End"][i])))
                
            mc_paths["Payoff Put"] = values_put

            def euro_vanilla(S, K, T, r, sigma, option = 'call'):
                
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
                    
                return result

            df_cost_of_hedge = pd.DataFrame([["Vanilla put", euro_vanilla(P, P_put_new, T, r, sigma, 'put'), x_min_new, x_min_new*euro_vanilla(P, P_put_new, T, r, sigma, 'put')]], 
                                    columns = ["Hedging instrument", "Theoretical price", "Number of instruments", "Theoretical hedging cost"])
            st.dataframe(df_cost_of_hedge)

            df_put_prices = pd.DataFrame([["Vanilla put", euro_vanilla(P, P_put_new, T, r, sigma, 'put'), np.exp(-r*T)*np.mean(values_put)]], 
                                    columns = ["Hedging instrument", "Analytical (Black-Scholes) price", "Monte Carlo estimate"])
            st.dataframe(df_put_prices)