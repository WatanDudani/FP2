#!/usr/bin/env python
# coding: utf-8

# In[9]:
def app():

    import streamlit as st
    import pandas as pd
    import numpy as np
    import urllib.request
    import json
    import plotly.express as px
    import matplotlib.pyplot as plt
    import seaborn as sns
    from PIL import Image
    import base64
    import altair as alt

    ## Basic setup and app layout
    #st.set_page_config(layout="wide")

    alt.renderers.set_embed_options(scaleFactor=2)
    
    img_width, img_height = 512, 512

    input_shape = (img_width, img_height)

    theme_image_name = 'image.jpg'
    logo_image = 'image.jpg'

    file_ = open(theme_image_name, "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    # rows = st.rows([1, 2])
    # # set image as logo
    # # st.sidebar.image(logo_image)
    # rows[0].image(logo_image)
    # rows[1].title('text image')
    # st.sidebar.title('Transforming oil production planning with real-time data, advanced algorithms, and actionable insights through our intuitive Oil Production Forecasting Application')

    # with st.sidebar:
    #     st.markdown("## Navigate to the Forecast tab to unlock precise oil production forecasting")
        # st.markdown("This is Final Review of FP2 Project at ISB")
        # st.markdown("## Welcome to the app on visualizing and forecasting Short Term and Long Term Forecasting of oil Production")
        # st.markdown("## We Hope you have fun while exploring##")
        # st.markdown("## Project Members are :")
        # st.markdown("Aditya Sarda - 12210016")
        # st.markdown("Komal Saini - 1221")
        # st.markdown("Tanisha Arora - 1221")
        # st.markdown("Watan Dudani - ")
        # st.markdown("Harshit Poddar - ")
        # st.markdown("Jaihind Yadav - ")

    st.title('Forecast with Confidence: Elevate Your Oil Production')
    # st.subheader('From Q3 1991 to Q1 2023')
    st.markdown("Data Source can be found [here](https://www.econdb.com/main-indicators?country=US&freq=M&tab=country-profile)")

    # st.subheader('This is the source file that we downloaded from the above mentioned link')


    @st.cache_data
    def read_raw ():
        URL = "https://raw.githubusercontent.com/WatanDudani/FP2/main/FP2_project-main/Source_data_oil_file/export.csv"
        df_oil_raw = pd.read_csv(URL, index_col=0)
        return df_oil_raw


    # In[13]:


    df_raw = read_raw()
    if df_raw not in st.session_state:
        st.session_state['df_raw'] = df_raw


    # In[14]:


    # st.write(df_raw)


    # In[15]:


    st.subheader('Final Clean and Preprocessed data used for forecasting')


    # In[16]:


    @st.cache_data
    def read_data_oil():
        URL = "https://raw.githubusercontent.com/WatanDudani/FP2/main/FP2_project-main/Source_data_oil_file/export.csv"
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
        df_oil_T['month'] = pd.to_datetime(df_oil_T['month']).dt.date 
        df_oil_T.set_index('month', inplace=True) 
        
        return df_oil_T

        


    # In[17]:


    df_a = read_data_oil()
    if df_a not in st.session_state:
        st.session_state['df_a'] = df_a


    # In[18]:


    st.write(df_a)


    # In[21]:

    st.subheader('Correlation Matrix used for evaluating the important features for forecasting')
    ##Correlation matrix
    fig, ax = plt.subplots()
    dataplot = sns.heatmap(df_a.corr(), cmap="YlGnBu",ax=ax, annot=True)
    st.write(fig)

