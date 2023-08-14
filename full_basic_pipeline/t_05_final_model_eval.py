import time

import joblib
import pandas as pd
from lazypredict.Supervised import LazyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support as score

# 1. Read in vectorized data
X_train_vect = pd.read_csv("C:\\Users\\danij\\Documents\\UC3M\\TFG\\DATA\\X_train_vect.csv")
X_test_vect = pd.read_csv("C:\\Users\\danij\\Documents\\UC3M\\TFG\\DATA\\X_test_vect.csv")

y_train = pd.read_csv("C:\\Users\\danij\\Documents\\UC3M\\TFG\\DATA\\train_labels.csv")
y_test = pd.read_csv("C:\\Users\\danij\\Documents\\UC3M\\TFG\\DATA\\test_labels.csv")

# --------------------------------------------------------------------------------------------------------------------

# 2. Create and train RF model

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

# --------------------------------------------------------------------------------------------------------------------

# 3. Make predictions with RF model

# Measuring the time taken to make predictions on the testing data
start = time.time()

# Making predictions
test_predictions = rf_model.predict(X_test_vect)

# Calculating total time taken to train the model
end = time.time()
pred_time = (end - start)

# --------------------------------------------------------------------------------------------------------------------

# 4. Evaluating RF model
# Computing precision, recall, fscore, and support values for the predicted results
precision, recall, fscore, support = score(y_test, test_predictions, average='macro')

# Printing the precision, recall, and F1-score
print('Macro Average Precision:', precision)
print('Macro Average Recall:', recall)
print('Macro Average F1-score:', fscore)
print('Fit time:', fit_time)
print('Predict time:', pred_time)

# --------------------------------------------------------------------------------------------------------------------

# 5. Evaluating other possible models

clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train_vect, X_test_vect, y_train, y_test)
print(models)

# --------------------------------------------------------------------------------------------------------------------

# 6. Saving predictions

# Create a new dataframe for submission
submission = pd.DataFrame({'tweet_id': X_test_vect['tweet_id'], 'label': test_predictions})

# Save the submission dataframe to a txt file
submission.to_csv('C:\\Users\\danij\\Documents\\UC3M\\TFG\\PREDICTIONS\\answer.txt', mode='w', sep='\t', index=False)
