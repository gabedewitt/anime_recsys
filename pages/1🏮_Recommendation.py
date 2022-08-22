import streamlit as st
import pandas as pd
from skimage import io
import matplotlib.pyplot as plt
def show_Cover(url):
    a = io.imread(url)
    plt.imshow(a)
    plt.axis('off')
    plt.show()

def data_frame_demo():
    @st.experimental_memo
    def get_Anime_data():
        df = pd.read_csv('./myanimelist/anime.csv')
        return df

    @st.experimental_memo
    def preprocess(dataframe):
        columns = ['title', 'type', 'score', 'scored_by', 'status', 'episodes', 'members',
           'favorites', 'rating', 'sfw', 'genres', 'themes', 'demographics',
           'studios', 'producers', 'licensors','synopsis']
        return dataframe[columns]

    df = get_Anime_data()
    df_pred = preprocess(df)
    anime_list = st.multiselect(
         "Choose some anime", list(df.title)
    )
    if not anime_list:
        st.error("Please select an anime.")
    else:
        df_subset = df[df["title"].isin(anime_list)]
        for picture in df_subset.main_picture:
            st.write(f'Anime selected: {df.title[df.main_picture == picture].tolist()[0]}')
            st.write(df[df.main_picture == picture].values.tolist())
            st.image(picture, caption = picture)
            


st.set_page_config(page_title="Recommendation", page_icon="🏮") 
st.markdown("# Anime Suggestion")
st.sidebar.header("Anime Suggestion")
st.write(
    """In this section choose an anime or a theme that you really like. The model will take care of the rest. Enjoy!"""
)

data_frame_demo()
