#!/usr/bin/env python
# coding: utf-8

# In[36]:

def forecast():

    import streamlit as st
    import pandas as pd
    import numpy as np
    import urllib.request
    import json
    import plotly.express as px
    import matplotlib.pyplot as plt
    import seaborn as sns
    import statsmodels.api as sm
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    import altair as alt
    import plotly.graph_objects as go

    ## Basic setup and app layout

    alt.renderers.set_embed_options(scaleFactor=2)

    @st.cache_data
    def read_data_oil():
        URL = "https://raw.githubusercontent.com/Royce281993/FP2_project/main/Source_data_oil_file/export.csv"
        df_oil = pd.read_csv(URL, index_col=0)
        df_oil = df_oil.reset_index()
        df_oil_T = df_oil.set_index('indicator').T
        df_oil_T.reset_index(drop=False, inplace=True)
        df_oil_T.drop(df_oil_T.tail(3).index,inplace=True)
        df_oil_T = df_oil_T.fillna(method='ffill')
        df_oil_T = df_oil_T[1:]
        df_oil_T['month'] = pd.to_datetime(df_oil_T['index'], format='%b %y')
        df_oil_T['Oil production'] = df_oil_T['Oil production'].str.replace(" ","")
        
            #####Making them into float format
        df_oil_T['Real gross domestic product'] = df_oil_T['Real gross domestic product'].astype('float')/100
        df_oil_T['Gross domestic product'] = df_oil_T['Gross domestic product'].astype('float')/100
        df_oil_T['Consumer price index'] = df_oil_T['Consumer price index'].astype('float')/100
        df_oil_T['Unemployment'] = df_oil_T['Unemployment'].astype('float')/100
        df_oil_T['Retail trade'] = df_oil_T['Retail trade'].astype('float')/100
        df_oil_T['Industrial production'] = df_oil_T['Industrial production'].astype('float')/100
        df_oil_T['Government balance'] = df_oil_T['Government balance'].astype('float')/100
        df_oil_T['Government debt'] = df_oil_T['Government debt'].astype('float')/100
        df_oil_T['Current account balance'] = df_oil_T['Current account balance'].astype('float')/100
        df_oil_T['Net international investment position'] = df_oil_T['Net international investment position'].astype('float')/100
        df_oil_T['Long term yield'] = df_oil_T['Long term yield'].astype('float')/100
        df_oil_T['House price'] = df_oil_T['House price'].astype('float')/100
        df_oil_T['Population'] = df_oil_T['Population'].astype('float')/100
        df_oil_T['Oil production'] = df_oil_T['Oil production'].astype('float')
        
        df_oil_T = df_oil_T.drop(['index','Population','Real gross domestic product'], axis=1)
        df_oil_T['month'] = pd.to_datetime(df_oil_T['month']) 
        df_oil_T.set_index('month', inplace=True) 
        
        return df_oil_T

        


    # In[30]:


    df_b = read_data_oil()
    if df_b not in st.session_state:
        st.session_state['df_b'] = df_b


    # In[31]:


    # Split the data into training and testing sets
    train_size = int(0.8 * len(df_b))
    train_data = df_b[:train_size]
    test_data = df_b[train_size:]
    target_variable = 'Oil production'


    # In[32]:


    st.title('Forecast with Confidence: Elevate Your Oil Production')

    st.subheader('Please select the horizon for time series forecasting')



    col1, col2 = st.columns(2)


    # Define the options for selectbox
    options = ['Short Term', 'Long Term']

    # Create the selectbox
    target = col1.selectbox('Select your target', options)

    # Define the slider range based on the selected value
    if target == 'Short Term':
        min_value = 1
        max_value = 4
    else:
        min_value = 1
        max_value = 24

    # Create the slider
    horizon = col2.slider('Choose the horizon', min_value, max_value, value=1, step=1)
    forecast_btn = st.button('Forecast')


    # In[35]:


    if forecast_btn:
        if target == 'Short Term':
            order = (0, 2, 1)
            seasonal_order = (1, 0, 1, 12) 
            sarima_model = sm.tsa.SARIMAX(train_data[target_variable], order=order, seasonal_order=seasonal_order)
            sarima_results = sarima_model.fit()
            sarima_predictions = sarima_results.predict(start=test_data.index[0], end=test_data.index[-1])
            sarima_mape = np.mean(np.abs((sarima_predictions - test_data[target_variable]) / test_data[target_variable])) * 100
            start_date = pd.to_datetime('2023-04-01')
            end_date = start_date + pd.DateOffset(months=horizon)
            predictions_for_horizon_sar = sarima_results.predict(start=start_date, end=end_date, dynamic=True)

            st.subheader('MAPE of Short term forecasting: ' + str(round(sarima_mape, 3)))

            # # Plotting the time series chart
            # fig, ax = plt.subplots(figsize=(10, 6))
            # ax.plot(df_b.index, df_b[target_variable], label='Actual')
            # ax.plot(predictions_for_horizon_sar.index, predictions_for_horizon_sar, label='Forecast')
            # ax.set_xlabel('Date')
            # ax.set_ylabel(target_variable)
            # ax.set_title('Short Term Forecast')
            # ax.legend()
            # st.pyplot(fig)

            #Another Approach
            

            fig = go.Figure()

            # Plotting the actual values with blue color
            fig.add_trace(go.Scatter(x=df_b.index, y=df_b[target_variable], mode='lines', name='Actual', line=dict(color='blue')))

            # Plotting the forecasted values with red color
            fig.add_trace(go.Scatter(x=predictions_for_horizon_sar.index, y=predictions_for_horizon_sar, mode='lines', name='Forecast', line=dict(color='red')))

            st.subheader('Short Term Forecasting (1-4 Months)')
            # Updating the layout of the figure
            fig.update_layout(
                title='Short Term Forecast',
                xaxis_title='Date',
                yaxis_title=target_variable,
                hovermode='x',
                height=600,
                width=1000,
                font=dict(family='Arial', size=12, color='black'),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                showlegend=True,
                legend=dict(
                    x=0,
                    y=1,
                    bgcolor='rgba(255,255,255,0.7)',
                    bordercolor='rgba(0,0,0,0.3)',
                    borderwidth=1,
                    orientation='v',
                    font=dict(color='black')
                )
            )

            fig.update_xaxes(showline=True, linewidth=1, linecolor='rgba(0,0,0,0.3)')
            fig.update_yaxes(showline=True, linewidth=1, linecolor='rgba(0,0,0,0.3)')

            # Displaying the interactive plot
            st.plotly_chart(fig)



            # Creating a DataFrame with forecasted values
            forecast_table = pd.DataFrame({'Date': predictions_for_horizon_sar.index.date, 'Forecast': predictions_for_horizon_sar.values})
            st.subheader('Forecasted Values')
            st.table(forecast_table)
           
        else :
            holt_winters_model = ExponentialSmoothing(train_data[target_variable], trend='add', seasonal='add', seasonal_periods=12)
            holt_winters_results = holt_winters_model.fit()
            start_date = pd.to_datetime('2023-04-01')
            end_date = start_date + pd.DateOffset(months=horizon)
            holt_winters_predictions = holt_winters_results.predict(start=test_data.index[0], end=test_data.index[-1])
            holt_winters_mape = np.mean(np.abs((holt_winters_predictions - test_data[target_variable]) / test_data[target_variable]))* 100
            predictions_for_horizon_hw = holt_winters_results.predict(start=start_date, end=end_date)
            
            st.subheader('MAPE of Long term forecasting: ' + str(round(holt_winters_mape, 3)))

            # # Plotting the time series chart for long term forecasting
            # fig, ax = plt.subplots(figsize=(10, 6))
            # ax.plot(df_b.index, df_b[target_variable], label='Actual')
            # ax.plot(predictions_for_horizon_hw.index, predictions_for_horizon_hw, label='Forecast')
            # ax.set_xlabel('Date')
            # ax.set_ylabel(target_variable)
            # ax.set_title('Long Term Forecast')
            # ax.legend()
            # st.pyplot(fig)

            #Another Approach
            fig = go.Figure()

            # Plotting the actual values with blue color
            fig.add_trace(go.Scatter(x=df_b.index, y=df_b[target_variable], mode='lines', name='Actual', line=dict(color='blue')))

            # Plotting the forecasted values with red color
            fig.add_trace(go.Scatter(x=predictions_for_horizon_hw.index, y=predictions_for_horizon_hw, mode='lines', name='Forecast', line=dict(color='red')))

            st.subheader('Long Term Forecasting (1-24 Months)')
            # Updating the layout of the figure
            fig.update_layout(
                title='Long Term Forecast',
                xaxis_title='Date',
                yaxis_title=target_variable,
                hovermode='x',
                height=600,
                width=1000,
                font=dict(family='Arial', size=12, color='black'),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                showlegend=True,
                legend=dict(
                    x=0,
                    y=1,
                    bgcolor='rgba(255,255,255,0.7)',
                    bordercolor='rgba(0,0,0,0.3)',
                    borderwidth=1,
                    orientation='v',
                    font=dict(color='black')
                )
            )

            fig.update_xaxes(showline=True, linewidth=1, linecolor='rgba(0,0,0,0.3)')
            fig.update_yaxes(showline=True, linewidth=1, linecolor='rgba(0,0,0,0.3)')

            # Displaying the interactive plot
            st.plotly_chart(fig)

            # Creating a DataFrame with forecasted values
            forecast_table = pd.DataFrame({'Date': predictions_for_horizon_hw.index.date, 'Forecast': predictions_for_horizon_hw.values})
            forecast_table['Forecast'] = forecast_table['Forecast'].round(2)  # Limit digits to 2 after the decimal point
            st.subheader('Forecasted Values')
            st.table(forecast_table)
