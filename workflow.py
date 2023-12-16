"""Workflow for the sentiment analysis of the McDonald's reviews."""

import re

import pandas as pd
import requests
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from textblob import TextBlob, Word


def get_reviews() -> list:
    """Get reviews from the McDonald's Yelp page."""
    links = [
        f"https://www.yelp.com/biz/mcdonalds-los-angeles-106?start={10+x*10}"
        for x in range(13)
    ]
    links.insert(0, "https://www.yelp.com/biz/mcdonalds-los-angeles-106")
    regex = re.compile("raw__")

    reviews = []
    for link in links:
        r = requests.get(link, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        results = soup.find_all("span", {"lang": "en"}, class_=regex)
        reviews = [*reviews, *[result.text for result in results]]

    return reviews


def preprocess(reviews: list) -> pd.DataFrame:
    """Preprocess the collected reviews."""
    df = pd.DataFrame(reviews, columns=["reviews"])
    stop_words = stopwords.words("english")

    # Lowercase
    df["review_lower"] = df["reviews"].str.lower()
    # Strip punctuation
    df["review_nopunc"] = df["review_lower"].str.replace("[^\w\s]", "", regex=True)
    # Removing Stopwords
    df["review_nostop"] = df["review_nopunc"].apply(
        lambda x: " ".join(word for word in x.split() if word not in stop_words),
    )
    # Removing other stopwords
    other_stopwords = ["one", "get", "go", "im", "2", "thru", "tell", "says", "two"]

    df["review_noother"] = df["review_nostop"].apply(
        lambda x: " ".join(word for word in x.split() if word not in other_stopwords),
    )
    # Lemmatization
    df["cleaned_review"] = df["review_noother"].apply(
        lambda x: " ".join(Word(word).lemmatize() for word in x.split()),
    )
    return df


def calculate_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the sentiment of the reviews."""
    df["polarity"] = df["cleaned_review"].apply(lambda x: TextBlob(x).sentiment[0])
    df["subjectivity"] = df["cleaned_review"].apply(lambda x: TextBlob(x).sentiment[1])
    return df


if __name__ == "__main__":
    reviews = get_reviews()
    df = preprocess(reviews)
    sentiment_df = calculate_sentiment(df)
    sentiment_df.to_csv("data/sentiment_results.csv")
