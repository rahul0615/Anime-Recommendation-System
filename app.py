import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_lottie import st_lottie
import re

# ------------------ UI CONFIG ------------------
st.set_page_config(page_title="Anime Recommender ML", layout="wide")

def load_lottie(path):
    with open(path, "r") as f:
        return f.read()

st_lottie(load_lottie("animation.json"), height=250)

st.markdown("<h1 style='color:#ff4b4b'>🎌 Anime Recommendation System</h1>", unsafe_allow_html=True)

# ------------------ NLP CLEANING ------------------
def clean_text(text):
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text.lower()

# ------------------ FETCH DATA ------------------
@st.cache_data
def fetch_anime():
    anime_list = []
    for page in range(1, 6):
        url = "https://api.jikan.moe/v4/anime"
        res = requests.get(url, params={"page": page, "limit": 25})
        data = res.json()["data"]

        for anime in data:
            genres = " ".join([g["name"] for g in anime["genres"]])
            synopsis = anime["synopsis"] or ""
            anime_list.append({
                "title": anime["title"],
                "genres": genres,
                "description": clean_text(genres + " " + synopsis),
                "image": anime["images"]["jpg"]["image_url"],
                "score": anime["score"]
            })

    return pd.DataFrame(anime_list)

df = fetch_anime()

# ------------------ TF-IDF MODEL ------------------
tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_matrix = tfidf.fit_transform(df["description"])
content_similarity = cosine_similarity(tfidf_matrix)

# ------------------ USER RATINGS (COLLABORATIVE) ------------------
st.sidebar.header("⭐ Rate Anime")
user_ratings = {}

for anime in df["title"].sample(5):
    rating = st.sidebar.slider(anime, 0, 10, 0)
    if rating > 0:
        user_ratings[anime] = rating

# ------------------ COLLAB FILTER ------------------
def collaborative_recommend():
    if not user_ratings:
        return pd.DataFrame()

    indices = [df[df["title"] == a].index[0] for a in user_ratings]
    scores = np.mean(content_similarity[indices], axis=0)

    df["collab_score"] = scores
    return df.sort_values("collab_score", ascending=False).head(5)

# ------------------ CONTENT FILTER ------------------
def content_recommend(anime_name):
    idx = df[df["title"] == anime_name].index[0]
    scores = list(enumerate(content_similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]
    return df.iloc[[i[0] for i in scores]]

# ------------------ SEARCH & FILTER ------------------
search = st.text_input("🔍 Search Anime")
genre_filter = st.multiselect("🎭 Filter by Genre", sorted(set(" ".join(df["genres"]).split())))

filtered_df = df.copy()

if search:
    filtered_df = filtered_df[filtered_df["title"].str.contains(search, case=False)]

if genre_filter:
    filtered_df = filtered_df[filtered_df["genres"].apply(lambda x: any(g in x for g in genre_filter))]

anime_selected = st.selectbox("🎥 Select Anime", filtered_df["title"])

# ------------------ RECOMMEND BUTTON ------------------
if st.button("🚀 Get Recommendations"):
    st.subheader("✨ Content-Based Recommendations")
    content_results = content_recommend(anime_selected)

    for _, row in content_results.iterrows():
        col1, col2 = st.columns([1, 4])
        with col1:
            st.image(row["image"], width=120)
        with col2:
            st.markdown(f"### {row['title']}")
            st.write(f"⭐ Score: {row['score']}")

    st.subheader("🤝 Personalized Recommendations")
    collab_results = collaborative_recommend()

    for _, row in collab_results.iterrows():
        col1, col2 = st.columns([1, 4])
        with col1:
            st.image(row["image"], width=120)
        with col2:
            st.markdown(f"### {row['title']}")
            st.write(f"⭐ Score: {row['score']}")