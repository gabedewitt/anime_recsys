import streamlit as st
import pandas as pd
from skimage import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import itertools
def show_Cover(url):
    a = io.imread(url)
    plt.imshow(a)
    plt.axis('off')
    plt.show()

def make_cloud(data, column):
    list_column = data[column].tolist()
    list_column = list(itertools.chain(*list_column))
    strings = ' '.join(list_column)

    plt.figure(figsize=(20,10))
    wordcloud = WordCloud(max_words=100,background_color="white",width=800, height=400).generate(strings)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
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
    df_pred.fillna(value = 'Not Found in MAL', inplace=True)

    list_columns = ['genres','themes','demographics','studios'
               ,'producers','licensors']

    for column in list_columns:
        df_pred[column] = df_pred[column].str.replace("['']","")
        df_pred[column] = df_pred[column].str.replace("\[\]","")
        
    anime_list = st.multiselect(
         "Choose some anime", list(df.title)
    )
    st.dataframe(df.head())
    st.dataframe(df_pred.head())
    st.pyplot(make_cloud(df,'demographics'))
    if not anime_list:
        st.error("Please select an anime.")
    else:
        df_subset = df[df["title"].isin(anime_list)]
        for picture in df_subset.main_picture:
            st.write(f'Anime selected: {df.title[df.main_picture == picture].tolist()[0]}')
            st.dataframe(df_pred[df.main_picture == picture])
            st.image(picture, caption = picture)
            
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title="Recommendation", page_icon="🏮") 
st.markdown("# Anime Suggestion")
st.sidebar.header("Anime Suggestion")
st.write(
    """In this section choose an anime or a theme that you really like. The model will take care of the rest. Enjoy!"""
)

data_frame_demo()
