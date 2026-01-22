import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("fake_news_dataset.csv")

x = data["text"]
y = data["label"]

vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
x_vectorized = vectorizer.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x_vectorized, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

st.title("Fake News Detector Using AI & ML")

news_input = st.text_area("Enter News Text")

if st.button("Check News"):
    if news_input.strip() == "":
        st.warning("Please enter some news text")
    else:
        news_vector = vectorizer.transform([news_input])
        prediction = model.predict(news_vector)
        if prediction[0] == 0:
            st.error("Fake News")
        else:
            st.success("Real News")
