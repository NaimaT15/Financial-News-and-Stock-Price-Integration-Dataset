import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import matplotlib.pyplot as plt

import re


def eda_analyst_ratings(df):
    # Convert the date column to datetime, handling timezone info correctly
    df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce')

    # Print the data types after conversion
    print("\nData type of 'date' column after conversion:")
    print(df['date'].dtype)

    # Print the first few rows of the date column after conversion
    print("\nConverted 'date' column:")
    print(df['date'])

    # Drop rows with invalid dates
    df = df.dropna(subset=['date'])

    # 1. Basic statistics for textual lengths (like headline length)
    df['headline_length'] = df['headline'].apply(len)
    print(f"Descriptive Statistics for headline length:\n{df['headline_length'].describe()}\n")
    
    # 2. Count the number of articles per publisher
    publisher_counts = df['publisher'].value_counts()
    print(f"Article Counts by Publisher:\n{publisher_counts}\n")
    
    # 3. Analyze the publication dates
    publication_trends = df['date'].dt.date.value_counts().sort_index()
    print(f"Publication Trends over Time:\n{publication_trends}\n")
    
    # Returning the modified DataFrame for further analysis
    return df


# Ensure the necessary resources are downloaded
nltk.download('vader_lexicon')

def perform_sentiment_analysis(df):
    # Initialize VADER SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    
    # Apply sentiment analysis on the 'headline' column
    df['sentiment'] = df['headline'].apply(lambda x: sia.polarity_scores(x)['compound'])

    # Categorize the sentiment into positive, negative, or neutral
    df['sentiment_label'] = df['sentiment'].apply(lambda score: 'positive' if score > 0.05 else ('negative' if score < -0.05 else 'neutral'))

    print("Sentiment analysis complete. Here are some samples:")
    print(df[['headline', 'sentiment', 'sentiment_label']].head())

    return df


    
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def perform_topic_modeling(df, num_topics=5, num_words=10):
    # Initialize CountVectorizer for basic keyword extraction
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    dtm = vectorizer.fit_transform(df['headline'])
    
    # Fit LDA model to identify topics
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(dtm)
    
    # Extract topics
    print("Identified Topics:")
    for index, topic in enumerate(lda.components_):
        print(f"Topic #{index + 1}:")
        print([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-num_words:]])
    
    return lda, dtm, vectorizer


def analyze_publication_frequency(df):
    # Print the original data type and a few rows of the date column
    print("Original 'date' column data type and sample data:")
    print(df['date'].dtype)
    print(df['date'].head())

   # Convert the date column to datetime, handling timezone info correctly
    df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce')

    # Print the data types after conversion
    print("\nData type of 'date' column after conversion:")
    print(df['date'].dtype)

    # Print the first few rows of the date column after conversion
    print("\nConverted 'date' column:")
    print(df['date'])
    # Print the data type and some sample data to confirm the conversion
    print("\nData type of 'date' column after conversion:")
    print(df['date'].dtype)
    print("\nSample of 'date' column after conversion:")
    print(df['date'].head())

    # Check if any NaT values were created during conversion
    print("\nNumber of NaT values in 'date' column after conversion:")
    print(df['date'].isna().sum())

    # Drop any rows with NaT in the 'date' column
    df = df.dropna(subset=['date'])

    # Confirm no NaT values remain
    print("\nConfirming no NaT values remain in 'date' column:")
    print(df['date'].isna().sum())

    # Aggregate the data by date
    df['publication_date'] = df['date'].dt.date
    publication_counts = df['publication_date'].value_counts().sort_index()

    # Plot the time series of publication frequency
    plt.figure(figsize=(12, 6))
    publication_counts.plot(kind='line', title='Publication Frequency Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Articles Published')
    plt.grid(True)
    plt.show()

    # Return the publication counts for further analysis if needed
    return publication_counts


def analyze_publishers(df):
    # Count the number of articles per publisher
    publisher_counts = df['publisher'].value_counts()
    
    # Display the top 10 publishers
    print("Top 10 Publishers by Number of Articles:")
    print(publisher_counts.head(10))
    
    return publisher_counts


def analyze_email_domains(df):
    # Identify publishers that look like email addresses
    email_publishers = df[df['publisher'].str.contains('@', na=False)]
    
    # Extract domain names from email addresses
    email_publishers['domain'] = email_publishers['publisher'].apply(lambda x: re.findall(r'@([\w\.-]+)', x)[0])
    
    # Count the number of articles per domain
    domain_counts = email_publishers['domain'].value_counts()
    
    # Display the top 10 domains
    print("Top 10 Domains by Number of Articles (from email addresses):")
    print(domain_counts.head(10))
    
    return domain_counts
