# FULL PIPELINE FOR TRAINING A ML MODEL FOR SENTIMENT MULTI-CLASSIFICATION TASK

# Importing necessary libraries
import re
import string
import time
import zipfile

import emoji
import joblib
import nltk
import numpy as np
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split

# Download the nlkt tools
nltk.download('opinion_lexicon')
nltk.download('sentiwordnet')
nltk.download('vader_lexicon')
nltk.download('punkt')

# Read in data
train_data = pd.read_csv("C:\\Users\\danij\\Documents\\UC3M\\TFG\\DATA\\train.csv")
dev_data = pd.read_csv("C:\\Users\\danij\\Documents\\UC3M\\TFG\\DATA\\dev.csv")

# Concatenate the train and dev data
data = pd.concat([train_data, dev_data])

# 1. Clean the data
# Downloading the stopwords corpus from NLTK
# words like "the", "is", "and" that are commonly used and can be ignored
stopwords = nltk.corpus.stopwords.words('english')

# Creating a WordNet lemmatizer object from NLTK (used for lemmatizing words to their base form based on context)
wn = nltk.WordNetLemmatizer()


# Function to clean the text by removing punctuation, converting to lowercase, and lemmatizing words
def clean_text(text):
    # Removing punctuation characters from the text and converting it to lowercase
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    # Splitting the text into tokens (words) using regular expressions that match just words
    tokens = re.split('\W+', text)
    # Lemmatizing each word in the tokens list using the WordNet lemmatizer
    text = [wn.lemmatize(word) for word in tokens if word not in stopwords]
    # Returning the cleaned text
    return text


# Applying the function to the dataset and convert the 'cleaned_text' column from list to string
data['cleaned_text'] = data['text'].apply(lambda x: clean_text(x)).apply(' '.join)
data.to_csv("C:\\Users\\danij\\Documents\\UC3M\\TFG\\DATA\\cleaned_text.csv", index=False)

# 2. Feature Engineering

# Applying the 'count' function to the 'text' column and storing the result in
# a new 'body_len' column. Not counting whitespaces
data['body_len'] = data['text'].apply(lambda x: len(x) - x.count(" "))


# Function to count the percentage of punctuation characters in a given text
def count_punct(text):
    # Counting the number of punctuation characters in the text
    count = sum([1 for char in text if char in string.punctuation])
    # Calculating the percentage of punctuation characters (excluding spaces) in the text
    return round(count / (len(text) - text.count(" ")), 3) * 100


# Applying the 'count_punct' function to the 'body_text' column and storing the result in
# a new 'punct%' column
data['punct%'] = data['text'].apply(lambda x: count_punct(x))

# Calculate average word length
data['avg_word_length'] = data['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

# 'sia_positive_word_rate', 'neutral_score',  'positive_score'
# Stopword Count
data['stopword_count'] = data['text'].apply(lambda x: len([word for word in x if word in stopwords]))

# Calculate word count
word_count = data['text'].apply(lambda x: len(str(x).split()))

# Create a Sentiment Intensity Analyzer
sia = SentimentIntensityAnalyzer()


# Function to get sentiment intensity
def get_sentiment_intensity(text):
    sentiment = sia.polarity_scores(text)
    return sentiment['compound']


# Apply the function to the text column
data['sentiment_intensity'] = data['cleaned_text'].apply(get_sentiment_intensity)

# Sentiment Scores
sentiment_scores = data['cleaned_text'].apply(lambda x: sia.polarity_scores(x))

# Positive Sentiment Score
data['positive_sscore'] = sentiment_scores.apply(lambda x: x['pos'])

# Negative Sentiment Score
data['negative_sscore'] = sentiment_scores.apply(lambda x: x['neg'])

# Neutral Sentiment Score
data['neutral_score'] = sentiment_scores.apply(lambda x: x['neu'])

# Count of Positive Words
positive_words = [word for word, score in sia.lexicon.items() if score > 0]
sia_positive_word_count = data['cleaned_text'].apply(lambda x: len([word for word in x if word in positive_words]))

# Count of Negative Words
negative_words = [word for word, score in sia.lexicon.items() if score < 0]
sia_negative_word_count = data['cleaned_text'].apply(lambda x: len([word for word in x if word in negative_words]))

# Positive Word Rate
data['sia_positive_word_rate'] = sia_positive_word_count / word_count

# Negative Word Rate
data['sia_negative_word_rate'] = sia_negative_word_count / word_count


# Emoji features
def count_emojis(text):
    emoji_list = [c for c in text if c in emoji.UNICODE_EMOJI_ENGLISH]
    return len(emoji_list)


# Assume you have an emoji_sentiment_dict
# Load the data into a pandas DataFrame
emoji_df = pd.read_csv('path to emoji csv')

# Convert the DataFrame into a dictionary
emoji_sentiment_dict = pd.Series(emoji_df.sentiment.values, index=emoji_df.emoji).to_dict()


def emoji_sentiment(text, emoji__dictionary):
    emoji_list = [c for c in text if c in emoji.UNICODE_EMOJI_ENGLISH]
    sentiment_score = sum([emoji__dictionary.get(emoji, 0) for emoji in emoji_list])
    return sentiment_score


data['emoji_count'] = data['text'].apply(count_emojis)
data['emoji_sentiment'] = data['text'].apply(lambda x: emoji_sentiment(x, emoji_sentiment_dict))

# 3. Split into train and test

# The 'label' column is used as the target variable (y)
# The rest of the columns except 'tweet_id' & 'therapy' are used as the features (X)
# The test_size parameter is set to 0.2, which means 20% of the data will be used for testing

X_train, X_test, y_train, y_test = train_test_split(
    data[
        ['tweet_id', 'cleaned_text', 'avg_word_length', 'sia_positive_word_rate', 'sia_negative_word_rate',
         'neutral_score', 'stopword_count', 'body_len', 'sentiment_intensity', 'negative_sscore',
         'punct%', 'positive_score']
    ], data['label'], test_size=0.2)

# 4. Vectorizing

# Creating a TfidfVectorizer object with the analyzer parameter set to the clean_text function
tfidf_vect = TfidfVectorizer()

# Fitting the TfidfVectorizer on the 'text' column of the training set
tfidf_vect_fit = tfidf_vect.fit(X_train['cleaned_text'])

# Transforming the 'text' column of the training and testing sets into TF-IDF features
tfidf_train = tfidf_vect_fit.transform(X_train['cleaned_text'])
tfidf_test = tfidf_vect_fit.transform(X_test['cleaned_text'])

# Concatenating the features columns with the TF-IDF features of the training set
X_train_vect = pd.concat([X_train[
                              ['avg_word_length', 'sia_positive_word_rate', 'neutral_score', 'stopword_count',
                               'body_len',
                               'sentiment_intensity', 'compound_score', 'punct%', 'positive_score']
                          ].reset_index(drop=True), pd.DataFrame(tfidf_train.toarray())], axis=1)

# Concatenating the features columns with the TF-IDF features of the testing set
X_test_vect = pd.concat([X_test[
                             ['avg_word_length', 'sia_positive_word_rate', 'neutral_score', 'stopword_count',
                              'body_len',
                              'sentiment_intensity', 'compound_score', 'punct%', 'positive_score']
                         ].reset_index(drop=True), pd.DataFrame(tfidf_test.toarray())], axis=1)

# Displaying the head (first few rows) of the X_train_vect DataFrame
X_train_vect.head()

# Displaying the head (first few rows) of the X_train_vect DataFrame
X_test_vect.head()

# 5. Evaluating the model

# Creating a RandomForestClassifier object with specified parameters
rf = RandomForestClassifier(n_estimators=150, max_depth=None, n_jobs=-1)

# Measuring the time taken to fit (train) the RandomForestClassifier on the training data
start = time.time()

# Training the model
rf_model = rf.fit(X_train_vect, y_train)
# Save the model for future use
joblib.dump(rf_model, "C:\\Users\\danij\\Documents\\UC3M\\TFG\\MODELS\\rf_model.pkl")

# Calculating total time taken to train the model
end = time.time()
fit_time = (end - start)

# Measuring the time taken to make predictions on the testing data
start = time.time()

# Making predictions
test_predictions = rf_model.predict(X_test_vect)

# Calculating total time taken to train the model
end = time.time()
pred_time = (end - start)

# 6. Submission of predictions
# Create a new dataframe for submission
submission = pd.DataFrame({'tweet_id': X_test_vect['tweet_id'], 'label': test_predictions})

# Save the submission dataframe to a txt file
submission.to_csv('C:\\Users\\danij\\Documents\\UC3M\\TFG\\PREDICTIONS\\answer.txt', sep='\t', index=False)

# Zip the submission file
with zipfile.ZipFile('C:\\Users\\danij\\Documents\\UC3M\\TFG\\PREDICTIONS\\answer.zip', 'w') as zipf:
    zipf.write('C:\\Users\\danij\\Documents\\UC3M\\TFG\\PREDICTIONS\\answer.txt')

# Computing precision, recall, fscore, and support values for the predicted results
precision, recall, fscore, support = score(y_test, test_predictions, average='macro')

# Printing the precision, recall, and F1-score
print('Macro Average Precision:', precision)
print('Macro Average Recall:', recall)
print('Macro Average F1-score:', fscore)
print('Fit time:', fit_time)
print('Predict time:', pred_time)
