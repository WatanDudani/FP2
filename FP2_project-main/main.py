# from home import *
# from forecast import *
from home import *
from forecast import *
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

def main():
    from PIL import Image
    import base64
    img_width, img_height = 512, 512

    input_shape = (img_width, img_height)

    theme_image_name = 'image.jpg'
    logo_image = 'image.jpg'

    file_ = open(theme_image_name, "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    st.sidebar.image(logo_image)
    st.sidebar.title("Control Panel")
    
    
    page_section = st.sidebar.radio("Sections: ", ('Home', 'Forecast'))

    if (page_section == 'Home'):
        
        # st.set_page_config('forecast', page_icon="🔮")

        # st.title("working homepage")
        app()

    elif (page_section == 'Forecast'):
        forecast() 


main()    