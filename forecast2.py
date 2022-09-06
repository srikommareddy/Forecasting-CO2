
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pandas.tseries.offsets import DateOffset
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
import warnings
import streamlit as st


warnings.filterwarnings("ignore")
co2data = pd.read_csv("co2dataset.csv")

# converting string data to datetime
co2data['Year'] = pd.to_datetime(co2data['Year'],format='%Y', errors='ignore')
co2data = co2data.set_index('Year')

co2data1 = co2data[70:]

#plt.figure(figsize=(12,4))
#sns.lineplot(x='Year', y= 'CO2', data = co2data1)

# loading the trained model
pickle_in = open('model_arima_712.pkl', 'rb') 
model = pickle.load(pickle_in)
 
train = co2data1[:101]
test = co2data1[101:]

df1 = co2data1
from pandas.tseries.offsets import DateOffset
future_dates=[df1.index[-1]+ DateOffset(years=x)for x in range(0,21)]
future_datest_df=pd.DataFrame(index=future_dates[1:],columns=df1.columns)
future_datest_df.tail()
future_df_ARIMA=pd.concat([df1,future_datest_df])


def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Streamlit CO2 Forecasting ML App</h1> 
    </div> 
    """
      
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
    st.line_chart(co2data1)
    if st.button("Get 20 years Forecast"):
        future_df_ARIMA['forecast_ARIMA']= model.predict(start = future_df_ARIMA.index[145], end = 165)   
        fpred = pd.DataFrame(future_df_ARIMA["forecast_ARIMA"][145:])
        st.dataframe(fpred)
        st.line_chart(future_df_ARIMA)
        
    
    
if __name__=='__main__': 
    main()


