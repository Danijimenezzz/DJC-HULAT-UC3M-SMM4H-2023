import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# 1. Read in data with new features
data = pd.read_csv("C:\\Users\\danij\\Documents\\UC3M\\TFG\\DATA\\data_cleaned_features.csv")
data.head()

# 2. Convert label to numeric
sentiment_label = {'neutral': 0, 'positive': 1, 'negative': 2}
data['num_label'] = data['label'].map(sentiment_label)
data.head()

# 3. Split into train, validation, and test set
# Divide variables into features and labels
features = data[['avg_word_length', 'sia_positive_word_rate', 'neutral_score', 'stopword_count', 'body_len',
                 'sentiment_intensity', 'compound_score', 'punct%', 'positive_score']]

labels = data['num_label']

# First split into train(60%) and test(40%), as we can only split dataset into 2
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=42)

# Now split test(40%) into test(20%) and validation(20%)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# Check if we splitted correctly
for dataset in [y_train, y_val, y_test]:
    print(round(len(dataset) / len(labels), 2))


# 4. Perform GridSearchCV
# Function to show performance of hyperparameters
def print_results(results):
    # Print the best parameters found during the cross-validation
    print('BEST PARAMS: {}\n'.format(results.best_params_))

    # Extract the mean and standard deviation of the test scores from the results object
    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']

    # Iterate over the means, stds, and params simultaneously using the zip function
    for mean, std, params in zip(means, stds, results.cv_results_['params']):
        # Print the mean score, the range (mean +/- 2 * std), and the corresponding parameters
        print('{} (+/-{}) for {}'.format(round(mean, 3), round(std * 2, 3), params))


# Try different models and see their performance
# Create an instance of the Random Forest Classifier
rf = RandomForestClassifier()

# Define the parameters to be tuned in the grid search
parameters = {
    'n_estimators': [5, 50, 100, 200],
    'max_depth': [2, 10, 20, None]
}

# Create an instance of GridSearchCV with the Random Forest Classifier and parameter grid
cv = GridSearchCV(rf, parameters, cv=5)

# Fit the training features and labels to the grid search cross-validation
cv.fit(X_train, y_train.values.ravel())

# Print the results of the grid search cross-validation
print_results(cv)

# Show best estimator
cv.best_estimator_
