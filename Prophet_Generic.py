# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 13:58:12 2019

@author: 611840750
"""
 
from fbprophet import Prophet
import numpy as np
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
from datetime import datetime as dt
from math import sqrt
import numpy as np
from operator import add
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from datetime import datetime, timedelta

#Reading the file
df = pd.read_csv('C:/Python/Prophet/sheet2.csv')
#Reading the holidays list

#Formatting the date
df['Process Date'] =  pd.to_datetime(df['Process Date'], format='%m/%d/%Y')

df = df.groupby(['Their PMN (TADIG) Code', 'Process Date'])['Total Charged Volume (MB)'].agg('sum')

df = df.to_frame().reset_index()
df = df.rename(columns={ 'Process Date':'ProcessDate', 'Total Charged Volume (MB)': 'TotalVolume', 'Their PMN (TADIG) Code': 'TADIG'})

#Loop in the customers
customers = ['NZLNH']
for x in customers:
    new_df = df.loc[df['TADIG'] == x]
    new_df.sort_values('ProcessDate', inplace=True) # Sort the data by ProcessDate
    new_df.reset_index(drop=True, inplace=True) # Reset the index
    new_df['ProcessDate'] = pd.to_datetime(new_df['ProcessDate'])
    
    final_df = pd.DataFrame(new_df, columns = ['ProcessDate', 'TotalVolume'])
    final_df = final_df.groupby('ProcessDate')['TotalVolume'].sum()
    final_df = pd.DataFrame(final_df)
    final_df = final_df.reset_index()
    final_df = final_df[final_df.TotalVolume != 0]
   
    final_df = final_df.rename(columns={ 'ProcessDate':'ds', 'TotalVolume': 'y'})
    
    final_df['y_orig'] = final_df['y'] # to save a copy of the original data..you'll see why shortly. 
    # log-transform y
    final_df['y'] = np.log(final_df['y'])
    
    #changepoint_prior_scale can be scaled as per the user requirements
    model = Prophet(changepoint_prior_scale=2.5)
    
    model.fit(final_df)
    
    
    ''' 'year': 'A',
            'quarter': 'Q',
            'month': 'M',
            'day': 'D',l
            'hour': 'H',
            'minute': 'T',
            'second': 'S',
            'millisecond': 'L',
            'microsecond': 'U',
            'nanosecond': 'N'}
    '''
    
    #make_future_dataframe predicts as per the periods and freq given, D is for Day wise prediction, 6 means our model will predict for the next 6 days.
    future_data = model.make_future_dataframe(periods=2, freq = 'D')
    
    forecast_data = model.predict(future_data)
    
    forecast_data[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10)
    
    forecast_data_orig = forecast_data # make sure we save the original forecast data
    forecast_data_orig['yhat'] = np.exp(forecast_data_orig['yhat'])
    forecast_data_orig['yhat_lower'] = np.exp(forecast_data_orig['yhat_lower'])
    forecast_data_orig['yhat_upper'] = np.exp(forecast_data_orig['yhat_upper'])
    
    #model.plot(forecast_data_orig)
    
    final_df['y_log']=final_df['y'] #copy the log-transformed data to another column
    final_df['y']=final_df['y_orig']
    
    
    final2_df = pd.DataFrame(forecast_data_orig)
    #latest_df = pd.concat([final2_df, final_df])
    latest_df = pd.merge(final2_df, final_df, on='ds')
    forecast_df = final2_df.iloc[len(final_df):]
    ult_df = pd.concat([latest_df, forecast_df])
    ult_df = pd.DataFrame(ult_df, columns = ['ds', 'y_orig', 'yhat', 'yhat_lower', 'yhat_upper'])
    #x_df = pd.DataFrame(latest_df, columns = ['ds', 'y_orig', 'yhat'])
    #model.plot(x_df)
    # plot
    pyplot.plot(ult_df['y_orig'], color = 'blue')   
    pyplot.plot(ult_df['yhat'], color='magenta')
    pyplot.show()
    
    text = "C:/Python/Raman_Prophet_Task/"
    final_text = text + str(x) + "_final.csv"
    ult_df.to_csv(final_text)
    
    ###################################################################################################
    ###################################################################################################
    ###################################################################################################
    ###################################################################################################
    #COMMENT THIS CODE IF YOU ARE RUNNING THE FLOW FOR MULTIPLE CUSTOMERS, AS THIS OFFLINE FUNCTIONALITY CREATES A TEMPORARY .html FILE FOR EACH CUSTOMER, HENCE RUNNING A CUSTOMER LOOP WON'T WORK FOR THIS PARTICULAR CODE
    ###################################################################################################
    ###################################################################################################
    ###################################################################################################
    ###################################################################################################
    import plotly.graph_objs as go
    import plotly.offline as py
    #Plot predicted and actual line graph with X=dates, Y=Outbound
    actual_chart = go.Scatter(y=ult_df["y_orig"], name= 'Actual')
    predict_chart = go.Scatter(y=ult_df["yhat"], name= 'Predicted')
    #predict_chart_upper = go.Scatter(y=ult_df["yhat_upper"], name= 'Predicted Upper')
    #predict_chart_lower = go.Scatter(y=ult_df["yhat_lower"], name= 'Predicted Lower')
    py.plot([actual_chart, predict_chart])


    
