# Importing necessary libraries
import string

import nltk
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Download the nlkt tools
nltk.download('opinion_lexicon')
nltk.download('sentiwordnet')
nltk.download('vader_lexicon')
nltk.download('punkt')

# 1. Read in cleaned data
data = pd.read_csv("C:\\Users\\danij\\Documents\\UC3M\\TFG\\DATA\\cleaned_text.csv")
test_data = pd.read_csv("C:\\Users\\danij\\Documents\\UC3M\\TFG\\DATA\\test_cleaned_text.csv")

# Show cleaned dataset
test_data.head()

# --------------------------------------------------------------------------------------------------------------------

# 2. Create new features, for dev and test data

# Applying the 'count' function to the 'text' column and storing the result in
# a new 'body_len' column. Not counting whitespaces
data['body_len'] = data['text'].apply(lambda x: len(x) - x.count(" "))
test_data['body_len'] = test_data['text'].apply(lambda x: len(x) - x.count(" "))


# Function to count the percentage of punctuation characters in a given text
def count_punct(text):
    # Counting the number of punctuation characters in the text
    count = sum([1 for char in text if char in string.punctuation])
    # Calculating the percentage of punctuation characters (excluding spaces) in the text
    return round(count / (len(text) - text.count(" ")), 3) * 100


# Applying the 'count_punct' function to the 'body_text' column and storing the result in
# a new 'punct%' column
data['punct%'] = data['text'].apply(lambda x: count_punct(x))
test_data['punct%'] = test_data['text'].apply(lambda x: count_punct(x))

# Create a Sentiment Intensity Analyzer
sia = SentimentIntensityAnalyzer()


# Function to get sentiment intensity
def get_sentiment_intensity(text):
    sentiment = sia.polarity_scores(text)
    return sentiment['compound']


# Apply the function to the text column
data['sentiment_intensity'] = data['cleaned_text'].apply(get_sentiment_intensity)
test_data['sentiment_intensity'] = test_data['cleaned_text'].apply(get_sentiment_intensity)

# Calculate word count
data['word_count'] = data['text'].apply(lambda x: len(str(x).split()))
test_data['word_count'] = test_data['text'].apply(lambda x: len(str(x).split()))

# Calculate character count
data['char_count'] = data['text'].apply(lambda x: len(str(x)))
test_data['char_count'] = test_data['text'].apply(lambda x: len(str(x)))

# Calculate average word length
data['avg_word_length'] = data['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
test_data['avg_word_length'] = test_data['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

# Calculate punctuation count
data['punctuation_count'] = data['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
test_data['punctuation_count'] = test_data['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

# Calculate hashtag count
data['hashtag_count'] = data['text'].apply(lambda x: len([c for c in str(x) if c == '#']))
test_data['hashtag_count'] = test_data['text'].apply(lambda x: len([c for c in str(x) if c == '#']))

# Stopword Count
stopwords = nltk.corpus.stopwords.words('english')
data['stopword_count'] = data['text'].apply(lambda x: len([word for word in x if word in stopwords]))
test_data['stopword_count'] = test_data['text'].apply(lambda x: len([word for word in x if word in stopwords]))

# Count of Positive Words
positive_words = [word for word, score in sia.lexicon.items() if score > 0]
data['sia_positive_word_count'] = data['cleaned_text'].apply(
    lambda x: len([word for word in x if word in positive_words]))
test_data['sia_positive_word_count'] = test_data['cleaned_text'].apply(
    lambda x: len([word for word in x if word in positive_words]))

# Count of Negative Words
negative_words = [word for word, score in sia.lexicon.items() if score < 0]
data['sia_negative_word_count'] = data['cleaned_text'].apply(
    lambda x: len([word for word in x if word in negative_words]))
test_data['sia_negative_word_count'] = test_data['cleaned_text'].apply(
    lambda x: len([word for word in x if word in negative_words]))

# Positive Word Rate
data['sia_positive_word_rate'] = data['sia_positive_word_count'] / data['word_count']
test_data['sia_positive_word_rate'] = test_data['sia_positive_word_count'] / test_data['word_count']

# Negative Word Rate
data['sia_negative_word_rate'] = data['sia_negative_word_count'] / data['word_count']
test_data['sia_negative_word_rate'] = test_data['sia_negative_word_count'] / test_data['word_count']

# Sentiment Intensity Analyzer
sia = SentimentIntensityAnalyzer()

# Sentiment Scores
sentiment_scores = data['cleaned_text'].apply(lambda x: sia.polarity_scores(x))
test_sentiment_scores = test_data['cleaned_text'].apply(lambda x: sia.polarity_scores(x))

# Positive Sentiment Score
data['positive_score'] = sentiment_scores.apply(lambda x: x['pos'])
test_data['positive_score'] = test_sentiment_scores.apply(lambda x: x['pos'])

# Negative Sentiment Score
data['negative_score'] = sentiment_scores.apply(lambda x: x['neg'])
test_data['negative_score'] = test_sentiment_scores.apply(lambda x: x['neg'])

# Neutral Sentiment Score
data['neutral_score'] = sentiment_scores.apply(lambda x: x['neu'])
test_data['neutral_score'] = test_sentiment_scores.apply(lambda x: x['neu'])

# Compound Sentiment Score
data['compound_score'] = sentiment_scores.apply(lambda x: x['compound'])
test_data['compound_score'] = test_sentiment_scores.apply(lambda x: x['compound'])

# Count of Laughing Expressions
laugh_expressions = ['haha', 'hehe', 'lol']
data['laugh_count'] = data['text'].apply(lambda x: sum([x.lower().count(expr) for expr in laugh_expressions]))

# Count of Sad Expressions
sad_expressions = [':(', ':-(', ';(', ';-(']
data['sad_count'] = data['text'].apply(lambda x: sum([x.count(expr) for expr in sad_expressions]))

# Initialize the VADER sentiment intensity analyzer
sid = SentimentIntensityAnalyzer()


# Define a function to calculate VADER sentiment scores
def get_vader_scores(text):
    scores = sid.polarity_scores(text)
    return scores


# Calculate VADER sentiment scores for each tweet
vader_scores = data['cleaned_text'].apply(get_vader_scores)

# Extract compound score for each tweet
data['compound_Vscore'] = vader_scores.apply(lambda x: x['compound'])

# Extract negative score for each tweet
data['negative_Vscore'] = vader_scores.apply(lambda x: x['neg'])

# Extract neutral score for each tweet
data['neutral_Vscore'] = vader_scores.apply(lambda x: x['neu'])

# Extract positive score for each tweet
data['positive_Vscore'] = vader_scores.apply(lambda x: x['pos'])

# --------------------------------------------------------------------------------------------------------------------

# 3. Showing new features
# Show all columns
pd.set_option('display.max_columns', None)

# Assuming 'column_to_drop' is the name of the column you want to drop
# data = data.drop('sentiment_scores', axis=1)

data.head()

# Store this dataframe with all of the features
data.to_csv("C:\\Users\\danij\\Documents\\UC3M\\TFG\\DATA\\all_features.csv", index=False)
test_data.to_csv("C:\\Users\\danij\\Documents\\UC3M\\TFG\\DATA\\test_all_features.csv", index=False)

# --------------------------------------------------------------------------------------------------------------------

# 4. Check feature importance to keep only most relevant features to the model
# Define the feature columns and the target column
feature_columns = ['body_len', 'punct%', 'sentiment_intensity', 'word_count', 'char_count', 'avg_word_length',
                   'punctuation_count', 'hashtag_count', 'stopword_count', 'sia_positive_word_count',
                   'sia_negative_word_count', 'sia_positive_word_rate', 'sia_negative_word_rate',
                   'positive_score', 'negative_score', 'neutral_score', 'compound_score', 'laugh_count', 'sad_count',
                   'compound_Vscore', 'negative_Vscore', 'neutral_Vscore', 'positive_Vscore']

target_column = 'label'

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(data[feature_columns], data[target_column], test_size=0.2,
                                                    random_state=42)

# Initialize a Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf.fit(X_train, y_train)

# Predict the sentiment labels on the test data
y_pred = rf.predict(X_test)

# Print the classification report
print(classification_report(y_test, y_pred))

# Extract the feature importance
feature_importance = rf.feature_importances_

# Create a DataFrame with the feature names and their importance scores
feature_importance_df = pd.DataFrame({'Feature': feature_columns, 'Importance': feature_importance})

# Sort the DataFrame by importance score in descending order
feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

feature_importance_df.head()

# Another way to evaluate features
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Initialize an array to store the feature importances
feature_importances = np.zeros(len(feature_columns))

# Number of iterations
n_iterations = 200

# Train the model multiple times with different random states
for i in range(n_iterations):
    rf = RandomForestClassifier(n_estimators=100, random_state=i)
    rf.fit(X_train, y_train)
    feature_importances += rf.feature_importances_

# Average the feature importances
feature_importances /= n_iterations

# Create a DataFrame with the feature names and their importance scores
feature_importance_df = pd.DataFrame({'Feature': feature_columns, 'Importance': feature_importances})

# Sort the DataFrame by importance score in descending order
feature_importance_df_2 = feature_importance_df.sort_values('Importance', ascending=False)
feature_importance_df_2.head()

# Now with n estimators
from sklearn.inspection import permutation_importance

# Train the model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Compute permutation importance
result = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)

# Create a DataFrame with the feature names and their importance scores
permutation_importance_df = pd.DataFrame({'Feature': feature_columns, 'Importance': result.importances_mean})

# Sort the DataFrame by importance score in descending order
permutation_importance_df = permutation_importance_df.sort_values('Importance', ascending=False)

permutation_importance_df.head()

# --------------------------------------------------------------------------------------------------------------------

# Dataframe of the columns you want to keep
data_cleaned_features = data[
    ['tweet_id', 'therapy', 'label', 'cleaned_text',
     'avg_word_length', 'sia_positive_word_rate', 'sia_negative_word_rate', 'neutral_score', 'stopword_count',
     'body_len', 'compound_score', 'punct%', 'positive_score', 'negative_score', 'neutral_score']]

test_data_cleaned_features = test_data[
    ['tweet_id', 'therapy', 'label', 'cleaned_text',
     'avg_word_length', 'sia_positive_word_rate', 'sia_negative_word_rate', 'neutral_score', 'stopword_count',
     'body_len', 'compound_score', 'punct%', 'positive_score', 'negative_score', 'neutral_score']]

test_data_cleaned_features.head()

# --------------------------------------------------------------------------------------------------------------------

# Write it to our data files
data_cleaned_features.to_csv("C:\\Users\\danij\\Documents\\UC3M\\TFG\\DATA\\data_cleaned_features.csv", mode='w',
                             index=False)

test_data_cleaned_features.to_csv("C:\\Users\\danij\\Documents\\UC3M\\TFG\\DATA\\test_data_cleaned_features.csv",
                                  mode='w', index=False)
