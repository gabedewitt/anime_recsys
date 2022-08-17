from skimage import io
import matplotlib.pyplot as plt
import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="Hello",
        page_icon="👋",
    )

    st.write("# Welcome to the Anime sugestion app! 👋")

    st.sidebar.success("Select an option above.")

    st.markdown(
        """
        This webapp offers a recommendation based on content information available 
        on the MyAnimeList website, such suggestions don't include series that have
        yet to air and some long running shows that do not have a known number of episodes.
        This means that sadly One Piece and Case Closed won't be recommended by this app,
        but both are definitely worth the reading if the prospect of 1000+ episodes feels
        too long for you.
        """
    )

    st.image("https://img1.ak.crunchyroll.com/i/spire4/9b3f967b806812e4b8ec9e8194e3a52a1658316525_main.jpg")



if __name__ == "__main__":
    run()