# Sentiment Analysis of Yelp Reviews

This project is focused on performing sentiment analysis on Yelp reviews, specifically for a McDonald's outlet in Los Angeles. It uses Python libraries such as BeautifulSoup for web scraping, NLTK for natural language processing, and TextBlob for sentiment analysis.

## Overview

The project aims to scrape reviews from Yelp, preprocess the text data, and then analyze the sentiment of each review. This analysis helps in understanding customer opinions and sentiments towards the services provided by the outlet.

## Features

- **Web Scraping**: Utilizes BeautifulSoup to scrape reviews from Yelp.
- **Data Preprocessing**: Includes text normalization, removing stopwords, and lemmatization.
- **Sentiment Analysis**: Uses TextBlob to calculate the polarity and subjectivity of each review.
- **Data Visualization**: Provides insights into common words and sentiment distribution.

## Installation

To set up the project, you need to install the required dependencies:

```bash
pip install requests beautifulsoup4 nltk textblob
```

## Usage

1. **Scraping Reviews**: Run the `workflow.py` script to scrape reviews from Yelp.
2. **Data Preprocessing**: The script preprocesses the scraped data for analysis.
3. **Sentiment Analysis**: The script then performs sentiment analysis on the preprocessed data.
4. **Results**: The sentiment analysis results are saved in `data/sentiment_results.csv`.

## Contributing

Contributions to improve the project are welcome. Please feel free to fork the repository, make changes, and submit a pull request.