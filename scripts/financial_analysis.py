import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob


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
