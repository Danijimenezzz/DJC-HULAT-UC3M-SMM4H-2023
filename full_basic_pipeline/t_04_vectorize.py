import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Read in cleaned featured data
data = pd.read_csv("C:\\Users\\danij\\Documents\\UC3M\\TFG\\DATA\\data_cleaned_features.csv")
test_data = pd.read_csv("C:\\Users\\danij\\Documents\\UC3M\\TFG\\DATA\\test_data_cleaned_features.csv")
test_data.head()

# --------------------------------------------------------------------------------------------------------------------

# 2. Split into train/test

X_train = data[['tweet_id', 'therapy', 'cleaned_text',
                'avg_word_length', 'sia_positive_word_rate', 'sia_negative_word_rate',
                'neutral_score', 'stopword_count', 'body_len', 'compound_score',
                'punct%', 'positive_score', 'negative_score', 'neutral_score']]

X_test = test_data[['tweet_id', 'therapy', 'cleaned_text',
                    'avg_word_length', 'sia_positive_word_rate', 'sia_negative_word_rate',
                    'neutral_score', 'stopword_count', 'body_len', 'compound_score',
                    'punct%', 'positive_score', 'negative_score', 'neutral_score']]

y_train = data['label']
y_test = test_data['label']

# --------------------------------------------------------------------------------------------------------------------

# 3. TD-IDF Vectorizer

# Creating a TfidfVectorizer object with the analyzer parameter set to the clean_text function
tfidf_vect = TfidfVectorizer()

# Fitting the TfidfVectorizer on the 'text' column of the training set
tfidf_vect_fit = tfidf_vect.fit(X_train['cleaned_text'])

# Transforming the 'text' column of the training and testing sets into TF-IDF features
tfidf_train = tfidf_vect_fit.transform(X_train['cleaned_text'])
tfidf_test = tfidf_vect_fit.transform(X_test['cleaned_text'])

# Concatenating the features columns with the TF-IDF features of the training set
X_train_vect = pd.concat([X_train[
                              ['tweet_id', 'avg_word_length', 'sia_positive_word_rate', 'sia_negative_word_rate',
                               'neutral_score',
                               'stopword_count', 'body_len', 'compound_score', 'punct%', 'positive_score',
                               'negative_score', 'neutral_score']
                          ].reset_index(drop=True), pd.DataFrame(tfidf_train.toarray())], axis=1)

# Concatenating the features columns with the TF-IDF features of the testing set
X_test_vect = pd.concat([X_test[
                             ['tweet_id', 'avg_word_length', 'sia_positive_word_rate', 'sia_negative_word_rate',
                              'neutral_score',
                              'stopword_count', 'body_len', 'compound_score', 'punct%', 'positive_score',
                              'negative_score', 'neutral_score']
                         ].reset_index(drop=True), pd.DataFrame(tfidf_test.toarray())], axis=1)

# Displaying the head (first few rows) of the X_train_vect DataFrame
X_train_vect.head()

# --------------------------------------------------------------------------------------------------------------------

# 4. Save vectorized dataframe

X_train_vect.to_csv('C:\\Users\\danij\\Documents\\UC3M\\TFG\\DATA\\X_train_vect.csv', mode='w', index=False)
X_test_vect.to_csv('C:\\Users\\danij\\Documents\\UC3M\\TFG\\DATA\\X_test_vect.csv', mode='w', index=False)

y_train.to_csv('C:\\Users\\danij\\Documents\\UC3M\\TFG\\DATA\\train_labels.csv', mode='w', index=False)
y_test.to_csv('C:\\Users\\danij\\Documents\\UC3M\\TFG\\DATA\\test_labels.csv', mode='w', index=False)
