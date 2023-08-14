# Importing necessary libraries
import time
import zipfile

import joblib
import pandas as pd

# Your code here ...

# Load the saved model from the file
rf = joblib.load("C:\\Users\\danij\\Documents\\UC3M\\TFG\\MODELS\\rf_model.pkl")

# Load test dataset
test_data = pd.read_csv("C:\\Users\\danij\\Documents\\UC3M\\TFG\\DATA\\test.csv") # path to your test.csv
# test_data['cleaned_text'] = test_data['text'].apply(lambda x: clean_text(x)).apply(' '.join)
# test_data.to_csv("C:\\Users\\danij\\Documents\\UC3M\\TFG\\DATA\\cleaned_test.csv", index=False)

# Your code here to calculate features for the test_data ...

# Vectorizing the test data...
test_data_vect = "Vectorized test data"

# Predict the test data
test_predictions = rf.predict(test_data_vect)

# Create a new dataframe for submission
submission = pd.DataFrame({'tweet_id': test_data['tweet_id'], 'label': test_predictions})

# Save the submission dataframe to a txt file
submission.to_csv('answer.txt', sep='\t', index=False)

# Zip the submission file
with zipfile.ZipFile('answer.zip', 'w') as zipf:
    zipf.write('answer.txt')
