import streamlit as st
import pandas as pd
from skimage import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import itertools
import scipy
import numpy as np
import validators
from ast import literal_eval
from sklearn.metrics.pairwise import sigmoid_kernel
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

def show_Cover(url: str) -> None:
    a = io.imread(url)
    plt.imshow(a)
    plt.axis('off')
    plt.show()

@st.experimental_memo
def preprocess_lists(df: pd.DataFrame, column_list: list) -> pd.DataFrame:    
    for column in column_list:
        string = column + '_treated'
        df_hold = df.loc[:,column]
        df_hold = df_hold.apply(lambda x: literal_eval(x) if len(x) > 2 else [])
        df[string] = df_hold
        df.drop(column, axis = 1, inplace = True)
    return df

def gen_wordcloud(df: pd.DataFrame, column_name: str) -> None:
    list_wc = df[column_name].tolist()
    list_wc = list(itertools.chain(*list_wc))
    strings = ' '.join(list_wc)

    plt.figure(figsize=(10,10))
    wordcloud = WordCloud(max_words=100,background_color="white",width=800, height=400, min_font_size = 10).generate(strings)
    fig, ax = plt.subplots(figsize = (10, 10))
    ax.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    st.pyplot(fig) 

@st.cache    
def vect_Tfid(series: pd.Series) -> scipy.sparse.csr_matrix:
    tfv = TfidfVectorizer(min_df=3,  max_features=None,
                      analyzer='word',
                      ngram_range=(1, 3),
                      stop_words = 'english')
    return tfv.fit_transform(series)

def sim_score(df: pd.DataFrame, kernel: str = 'sigmoid') -> np.ndarray:
    tfv_matrix = vect_Tfid(df['synopsis'])
    if kernel == 'sigmoid':
        return sigmoid_kernel(tfv_matrix, tfv_matrix)
    elif kernel == 'linear':    
        return linear_kernel(tfv_matrix, tfv_matrix) 


@st.experimental_memo(persist = 'disk')
def get_rec(entry: str, df: pd.DataFrame, sug_num: int, rec_type: str) -> pd.DataFrame:
    idx = pd.Series(df.index, index=df['title']).drop_duplicates()[entry]

    df_sim = list(enumerate(sim_score(df, rec_type)[idx]))

    sim_scores = sorted(df_sim, key = lambda x: x[1], reverse = True)

    sim_recs = sim_scores[1:sug_num]

    anime_indices = [y[0] for y in sim_recs]

    return df['title'].iloc[anime_indices]

def data_frame_demo() -> None:
    @st.experimental_memo
    def get_Anime_data() -> None:
        df = pd.read_csv('./myanimelist/anime.csv')
        return df

    @st.experimental_memo
    def preprocess(dataframe: pd.DataFrame) -> pd.DataFrame:
        columns = ['title', 'type', 'score', 'scored_by', 'status', 'episodes', 'members',
           'favorites', 'rating', 'sfw', 'genres', 'themes', 'demographics',
           'studios', 'producers', 'licensors','synopsis']
        return dataframe[columns]

    df = get_Anime_data()
    df_pred = preprocess(df)
    df_pred.fillna(value = 'Not Found in MAL', inplace=True)

    list_columns = ['genres','themes','demographics','studios'
               ,'producers','licensors']

    df_pred = preprocess_lists(df_pred, list_columns)
        
    anime_list = st.multiselect(
         "Choose some anime", list(df.title)
    )
    #st.dataframe(df.head()) Used for testing
    #st.dataframe(df_pred.head()) Used for testing
    if not anime_list:
        st.error("Please select an anime.")
    else:
        df_subset = df[df["title"].isin(anime_list)]
        r_type = st.selectbox('Which kernel to be used for the recommendation?',
        ('sigmoid', 'linear'))
        rec_num = st.slider('How many recommendations?', 10, 50, 20)
        for anime, picture, url, trailer in zip(anime_list, df_subset.main_picture, df_subset.url, df_subset.trailer_url):
            col1, col2, col3 = st.columns([2,4,4])
            with col1:
                st.write(f'Anime selected: {anime}')
                #st.dataframe(df_subset) used for testing
                st.image(picture, caption = picture)
                st.write(f'[MAL page]({url})')
                if validators.url(trailer):
                    st.video(trailer)
            with col2:    
                rec_list = get_rec(anime, df_pred, rec_num, r_type)
                rec_df = df_pred[df_pred["title"].isin(rec_list)]
                st.dataframe(rec_df[['title','licensors_treated','sfw']], 
                height=550, width= 810)
            with col3:
                gen_wordcloud(rec_df,'genres_treated') 
                gen_wordcloud(rec_df,'themes_treated') 
         
            
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title="Recommendation", page_icon="🏮", layout="wide") 
st.markdown("# Anime Suggestion")
st.sidebar.header("Anime Suggestion")
st.write(
    """In this section choose an anime or a theme that you really like. The model will take care of the rest. Enjoy!"""
)

data_frame_demo()
