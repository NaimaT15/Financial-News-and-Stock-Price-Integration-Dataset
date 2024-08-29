import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import matplotlib.pyplot as plt
from gensim import corpora
from gensim.models import LdaModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re
import talib as ta
import pynance as pn
import pyfolio as pf


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



def perform_topic_modeling_gensim(df, num_topics=5, num_words=10, sample_size=None):
    # Subsample the data if a sample size is provided
    if sample_size:
        df = df.sample(n=sample_size, random_state=42)

    # Initialize CountVectorizer for basic keyword extraction
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    dtm = vectorizer.fit_transform(df['headline'])
    vocab = vectorizer.get_feature_names_out()

    # Convert to gensim format
    corpus = [[(i, freq) for i, freq in zip(doc.indices, doc.data)] for doc in dtm]
    id2word = corpora.Dictionary([vocab])

    # Fit LDA model using gensim
    lda_model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=id2word, passes=15, random_state=42)

    # Print topics
    topics = lda_model.print_topics(num_topics=num_topics, num_words=num_words)
    for topic in topics:
        print(topic)

    return lda_model, corpus, id2word


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

    # Plot the time series of publication frequency as a line plot
    plt.figure(figsize=(12, 6))
    plt.plot(publication_counts, linestyle='-', marker=None, color='b')
    plt.title('Publication Frequency Over Time')
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
def visualize_sentiment_distribution(df):
    # Count the occurrences of each sentiment category
    sentiment_counts = df['sentiment_label'].value_counts()
    
    # Plot a bar chart of the sentiment distribution
    plt.figure(figsize=(10, 6))
    sentiment_counts.plot(kind='bar', color=['green', 'red', 'blue'])
    plt.title('Distribution of News Sentiment')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=0)
    plt.show()

def visualize_sentiment_by_publisher(df, top_n=5):
    # Select the top N publishers
    top_publishers = df['publisher'].value_counts().head(top_n).index
    filtered_df = df[df['publisher'].isin(top_publishers)]
    
    # Group by publisher and sentiment label, then count
    sentiment_by_publisher = filtered_df.groupby(['publisher', 'sentiment_label']).size().unstack(fill_value=0)
    
    # Plot the sentiment distribution by publisher
    sentiment_by_publisher.plot(kind='bar', stacked=True, figsize=(12, 8))
    plt.title('Sentiment Distribution by Publisher')
    plt.xlabel('Publisher')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=45)
    plt.show()


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



# TASK 2
def load_stock_data(file_path):
    # Load the stock price data
    df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    
    # Ensure the data has the necessary columns
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(column in df.columns for column in required_columns):
        raise ValueError(f"Data must include the following columns: {required_columns}")
    
    
    return df



def apply_technical_indicators(df):
    # Calculate Simple Moving Average (SMA)
    df['SMA_20'] = ta.SMA(df['Close'], timeperiod=20)
    
    # Calculate Relative Strength Index (RSI)
    df['RSI_14'] = ta.RSI(df['Close'], timeperiod=14)
    
    # Calculate Moving Average Convergence Divergence (MACD)
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = ta.MACD(df['Close'], 
                                                              fastperiod=12, 
                                                              slowperiod=26, 
                                                              signalperiod=9)
    

    
    return df



def calculate_financial_metrics(df):
    # Example: Calculate daily returns using PyNance
    df['daily_return'] = df['Close'].pct_change()
    

    return df



def visualize_stock_data(df):
    plt.figure(figsize=(14, 7))
    
    # Plot Closing Price
    plt.subplot(3, 1, 1)
    plt.plot(df.index, df['Close'], label='Close Price')
    plt.plot(df.index, df['SMA_20'], label='20-Day SMA')
    plt.title('Stock Closing Price and SMA')
    plt.legend()
    
    # Plot RSI
    plt.subplot(3, 1, 2)
    plt.plot(df.index, df['RSI_14'], label='RSI (14)')
    plt.axhline(y=70, color='r', linestyle='--')
    plt.axhline(y=30, color='g', linestyle='--')
    plt.title('RSI (14)')
    plt.legend()
    
    # Plot MACD
    plt.subplot(3, 1, 3)
    plt.plot(df.index, df['MACD'], label='MACD')
    plt.plot(df.index, df['MACD_signal'], label='Signal Line')
    plt.bar(df.index, df['MACD_hist'], label='MACD Histogram')
    plt.title('MACD')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


