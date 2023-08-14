# Importing necessary libraries
import re
import string

import nltk
import pandas as pd

# Download the nlkt tools
nltk.download('opinion_lexicon')
nltk.download('sentiwordnet')

# 1. Read in raw data
train_data = pd.read_csv("C:\\Users\\danij\\Documents\\UC3M\\TFG\\DATA\\train.csv")
dev_data = pd.read_csv("C:\\Users\\danij\\Documents\\UC3M\\TFG\\DATA\\dev.csv")

# Concatenate the train and dev data
# data = pd.concat([train_data, dev_data])
data = train_data

# Test data (2 options, keep dev data as test, or import a new file without labels)
# test_data = pd.read_csv("C:\\Users\\danij\\Documents\\UC3M\\TFG\\DATA\\dev.csv")
test_data = dev_data

# Set the display options to show the full content of each row
pd.set_option('display.max_colwidth', -1)

# Show initial data
data.head()

# 2. Downloading the stopwords corpus from NLTK
# words like "the", "is", "and" that are commonly used and can be ignored
stopwords = nltk.corpus.stopwords.words('english')

# 3. Creating a WordNet lemmatizer object from NLTK (used for lemmatizing words to their base form based on context)
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


# 4. Applying the function to the train and test dataset and convert the 'cleaned_text' column from list to string
data['cleaned_text'] = data['text'].apply(lambda x: clean_text(x)).apply(' '.join)
test_data['cleaned_text'] = test_data['text'].apply(lambda x: clean_text(x)).apply(' '.join)

# Show the resulting, cleaned dataset
data.head()

# 5. Save cleaned dataset
data.to_csv("C:\\Users\\danij\\Documents\\UC3M\\TFG\\DATA\\cleaned_text.csv", mode='w', index=False)
test_data.to_csv("C:\\Users\\danij\\Documents\\UC3M\\TFG\\DATA\\test_cleaned_text.csv", mode='w', index=False)
