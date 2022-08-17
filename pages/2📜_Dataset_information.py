import streamlit as st
import time
import numpy as np

st.set_page_config(page_title="Dataset information", page_icon="📜") 


st.markdown(
        """
        # Dataset information

        This webapp is uses the MyAnimeList Anime and Manga Datasets from Andreu Vall Hernàndez
        available on Kaggle as such dataset has both info scraped with the official API and Jikan API. 
        Which makes it the best option available, since it's weekly updated and covers both anime and manga.
        
        Link to the dataset: <https://www.kaggle.com/datasets/andreuvallhernndez/myanimelist>
        """
    )
st.image('https://i.imgur.com/vEy5Zaq.png', width=300, caption = 'MyAnimeList Logo')
st.image('https://www.kaggle.com/static/images/logos/kaggle-logo-gray-300.png', width=300, caption = 'Kaggle Logo')