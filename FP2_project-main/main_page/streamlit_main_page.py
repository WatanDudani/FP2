#!/usr/bin/env python
# coding: utf-8

# In[9]:


import streamlit as st
import pandas as pd
import numpy as np
import urllib.request
import json
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns


# In[10]:


st.set_page_config('Homepage', page_icon="🏡")


# In[11]:


with st.sidebar:
    st.markdown("# Hello Professor")
    st.markdown("This is Final Review of FP2 Project at ISB")
    st.markdown("## Welcome to the app on visualizing and forecasting Short Term and Long Term Forecasting of oil Production")
    st.markdown("## We Hope you have fun while exploring##")
    st.markdown("## Project Members are :")
    st.markdown("Aditya Sarda - 12210016")
    st.markdown("Komal Saini - 1221")
    st.markdown("Tanisha Arora - 1221")
    st.markdown("Watan Dudani - ")
    st.markdown("Harshit Poddar - ")
    st.markdown("Jaihind Yadav - ")

st.title('Short/Long term forecasting for oil production')
st.subheader('From Q3 1991 to Q1 2023')
st.markdown("Source table can be found [here](https://www.econdb.com/main-indicators?country=US&freq=M&tab=country-profile)")

st.subheader('This is the source file that we downloaded from the above mentioned link')


# In[12]:


@st.cache_data
def read_raw ():
    URL = "https://raw.githubusercontent.com/Royce281993/FP2_project/main/Source_data_oil_file/export.csv"
    df_oil_raw = pd.read_csv(URL, index_col=0)
    return df_oil_raw


# In[13]:


df_raw = read_raw()
if df_raw not in st.session_state:
    st.session_state['df_raw'] = df_raw


# In[14]:


st.write(df_raw)


# In[15]:


st.subheader('After operations on the source file, the file which will be used for forecasting looks like this')


# In[16]:


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

    


# In[17]:


df_a = read_data_oil()
if df_a not in st.session_state:
    st.session_state['df_a'] = df_a


# In[18]:


st.write(df_a)


# In[21]:


##Correlation matrix
fig, ax = plt.subplots()
dataplot = sns.heatmap(df_a.corr(), cmap="YlGnBu",ax=ax, annot=True)
st.write(fig)

