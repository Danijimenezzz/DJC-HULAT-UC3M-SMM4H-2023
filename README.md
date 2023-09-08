# DJC-HULAT-UC3M-SMM4H-2023
DJC-HULAT-UC3M-SMM4H 2023 Classification of therapy sentiment in tweets

## References

For this project, other works were studied and reviewed, also somo pieces of code from this previous projects were re-used for our thesis.

The mentioned previous works are listed below (also, more projects were investigated for research purpouses, and are mentioned in the thesis paper:

- [Getting Started with Sentiment Analysis on Twitter](https://huggingface.co/blog/sentiment-analysis-twitter), by Federico Pascual

- [Twitter-Sentiment-Analysis-about-ChatGPT](https://github.com/hxycorn/Twitter-Sentiment-Analysis-about-ChatGPT/tree/main), by hxycorn

- [pytorch-sentiment-analysis](https://github.com/bentrevett/pytorch-sentiment-analysis), by bentrevett

- [twitter-sentiment-analysis](https://github.com/abdulfatir/twitter-sentiment-analysis), by abdulfatir

- [awesome-sentiment-analysis](https://github.com/xiamx/awesome-sentiment-analysis), by xiamx

- [Sentiment-Analysis-Twitter](https://github.com/ayushoriginal/Sentiment-Analysis-Twitter), by ayushoriginal

- [SentimentAnalysis](https://github.com/barissayil/SentimentAnalysis), by barissayil

- [Sentiment analysis using Twitter data: a comparative application of lexicon- and machine-learning-based approach](https://link.springer.com/article/10.1007/s13278-023-01030-x), by Yuxing Qi & Zahratu Shabrina

- [Twitter-Sentiment-Analysis](https://github.com/the-javapocalypse/Twitter-Sentiment-Analysis), by the-javapocalypse

- [french-sentiment-analysis-with-bert](https://github.com/TheophileBlard/french-sentiment-analysis-with-bert), by TheophileBlard

## Code Overview

This pipeline code is a step-by-step implementation to build and evaluate a machine learning model for sentiment classification of English tweets related to various therapies. The goal is to classify tweets into three sentiment classes: positive, negative, and neutral.

### Justification of chosen libraries 

1. **re (Regular Expressions) and string:**
   - Justification: These libraries are used to preprocess and clean text data by removing punctuation and special characters. Regular expressions are powerful tools for pattern matching, which helps clean and tokenize text data effectively.

2. **nltk (Natural Language Toolkit):**
   - Justification: NLTK provides various resources for natural language processing tasks. In this code, NLTK is used for text preprocessing, sentiment analysis, and word lemmatization. The SentimentIntensityAnalyzer from NLTK's VADER module is used for sentiment analysis.

3. **pandas:**
   - Justification: Pandas is used for data manipulation and analysis. It is particularly useful for reading and writing CSV files, which are used for storing and processing the dataset.

4. **numpy:**
   - Justification: NumPy is a fundamental package for scientific computing in Python. While not extensively used in the code, it can be used for various numerical operations and calculations.

5. **SentimentIntensityAnalyzer (from nltk.sentiment.vader):**
   - Justification: This class from NLTK's VADER module allows for sentiment analysis of text data. It assigns a polarity score to each text, indicating its positive, negative, or neutral sentiment.

6. **RandomForestClassifier (from sklearn.ensemble):**
   - Justification: RandomForestClassifier is a powerful ensemble learning method for classification tasks. It combines multiple decision trees to improve predictive accuracy and control overfitting.

7. **classification_report (from sklearn.metrics):**
   - Justification: This function generates a comprehensive classification report, including precision, recall, F1-score, and support for each class. It is used to evaluate the performance of the trained model.

8. **train_test_split (from sklearn.model_selection):**
   - Justification: This function is used to split the dataset into training and testing sets, enabling the model to be trained on one subset and evaluated on another. It helps assess the model's generalization ability.

9. **TfidfVectorizer (from sklearn.feature_extraction.text):**
   - Justification: TfidfVectorizer is used to convert text data into numerical features (TF-IDF vectors), which are essential for training machine learning models that require numeric input.

10. **time:**
    - Justification: The time module is used to measure the time taken for training and prediction steps. It helps to assess the efficiency of the model.

11. **joblib:**
    - Justification: Joblib is used to save and load trained machine learning models. It is particularly useful when dealing with large models that need to be stored and reused.

12. **zipfile:**
    - Justification: The zipfile module is used to create a compressed zip file containing the submission file for the competition. This is a common practice in data science competitions.

13. **LazyClassifier (from lazypredict.Supervised):**
    - Justification: LazyClassifier is a tool that quickly runs and compares various machine learning models on a dataset, providing an initial overview of model performances without manual parameter tuning.

Each library serves a specific purpose and contributes to different stages of the machine learning pipeline, from data preprocessing to model training, evaluation, and submission.

### Code Steps

**Data Preprocessing:**
1. The code starts by reading the training and testing data from CSV files and concatenating the train and dev data (if applicable).
2. It then downloads NLTK resources required for sentiment analysis, such as the Opinion Lexicon and SentiWordNet.
3. The provided text cleaning function `clean_text` is applied to each tweet's text, which removes punctuation, converts text to lowercase, tokenizes, and lemmatizes words. The cleaned text is stored in a new column named `cleaned_text`.
4. Additional features like word count, character count, sentiment intensity scores, etc., are calculated and added to the DataFrame.

**Feature Extraction and Vectorization:**
1. The code uses a TF-IDF vectorizer from `scikit-learn` to convert the cleaned text into numerical features.
2. The `TfidfVectorizer` is fit on the training data and transforms both training and testing text data into TF-IDF features.
3. The original features are concatenated with the TF-IDF features to create a feature-rich dataset.

**Model Training and Evaluation:**
1. A Random Forest Classifier from `scikit-learn` is used as the main sentiment classification model.
2. GridSearchCV is employed to perform hyperparameter tuning for the Random Forest model. It searches through different combinations of `n_estimators` and `max_depth` parameters to find the best model.
3. The trained model's performance is evaluated on the test dataset, calculating precision, recall, F1-score, and support for each class. The model's fit time and prediction time are also recorded.

**Alternative Model Evaluation:**
1. The `LazyClassifier` from `lazypredict` is used to quickly evaluate various machine learning models without manually specifying them.
2. The LazyClassifier outputs a summary of the performance of different models on the provided dataset.

**Model Prediction and Submission:**
1. The trained Random Forest model is used to predict the sentiment labels of the test data.
2. The predictions are saved in a submission DataFrame with tweet IDs and predicted labels.
3. The submission DataFrame is saved as a tab-separated text file named `answer.txt`.
4. Finally, the `zipfile` library is used to create a zip file named `answer.zip` containing the submission text file.

### Feature Engineering 

1. **body_len:** The length (number of characters) of the text body. This feature captures the overall length of the text, which can influence the complexity and content of the message.

2. **punct%:** The percentage of punctuation characters in the text body. Punctuation can provide insights into the writing style and emotional tone of the text.

3. **sentiment_intensity:** The sentiment intensity score obtained from the Sentiment Intensity Analyzer (SIA). This score indicates the overall sentiment of the text, whether it's positive, negative, or neutral.

4. **word_count:** The number of words in the text body. This feature provides information about the text's length and complexity.

5. **char_count:** The number of characters in the text body. Similar to `word_count`, this feature provides insights into the text's length.

6. **avg_word_length:** The average length of words in the text body. This feature can give an idea of the complexity of vocabulary used.

7. **punctuation_count:** The count of punctuation symbols in the text body. This helps in understanding the usage of punctuation for emphasis or expression.

8. **hashtag_count:** The count of hashtags (#) used in the text. Hashtags can signify topics or themes that are relevant to the text.

9. **stopword_count:** The count of stopwords (common words like "the," "and," "is") in the text. Stopwords are often filtered out during text analysis due to their limited semantic meaning.

10. **sia_positive_word_count:** The count of words with positive sentiment according to SIA. This feature indicates the presence of positive language in the text.

11. **sia_negative_word_count:** The count of words with negative sentiment according to SIA. This feature indicates the presence of negative language in the text.

12. **sia_positive_word_rate:** The ratio of positive sentiment words to the total words in the text. This feature gives an idea of the overall positive tone of the text.

13. **sia_negative_word_rate:** The ratio of negative sentiment words to the total words in the text. This feature gives an idea of the overall negative tone of the text.

14. **positive_score:** The positive sentiment score obtained from SIA. It quantifies the strength of positive sentiment in the text.

15. **negative_score:** The negative sentiment score obtained from SIA. It quantifies the strength of negative sentiment in the text.

16. **neutral_score:** The neutral sentiment score obtained from SIA. It quantifies the strength of neutral sentiment in the text.

17. **compound_score:** The compound sentiment score obtained from SIA. It represents the overall sentiment of the text, combining positive, negative, and neutral sentiments.

18. **laugh_count:** The count of laughing emoticons in the text. Emoticons can convey emotions and provide context to the text.

19. **sad_count:** The count of sad emoticons in the text. Similar to `laugh_count`, this feature captures emotional expression.

20. **compound_Vscore:** The compound Valence score obtained from the Valence Aware Dictionary and sEntiment Reasoner (VADER) lexicon. It's an alternative sentiment indicator.

21. **negative_Vscore:** The negative Valence score from VADER. This lexicon-based score measures negative sentiment.

22. **neutral_Vscore:** The neutral Valence score from VADER. This lexicon-based score measures neutral sentiment.

23. **positive_Vscore:** The positive Valence score from VADER. This lexicon-based score measures positive sentiment.

These features capture various aspects of the text, including its length, sentiment, emotional content, writing style, and vocabulary complexity. By including a combination of sentiment analysis and linguistic features, the model can learn patterns that might distinguish between different classes of text.


### Hyperparameter Tuning

```python
# Creating a RandomForestClassifier object with specified parameters
rf = RandomForestClassifier(n_estimators=150, max_depth=None, n_jobs=-1)
```

1. **n_estimators (Number of Trees):**
   - Value: 150
   - Rationale: This parameter determines the number of decision trees in the random forest ensemble. A higher number of trees can increase the model's predictive power and stability. However, a balance needs to be struck, as too many trees might lead to overfitting. The selected value, 150, is a reasonable compromise between model complexity and performance.

2. **max_depth (Maximum Depth of Trees):**
   - Value: None (Unlimited)
   - Rationale: The max_depth parameter controls the depth of individual decision trees in the forest. A deeper tree can capture more complex relationships in the data, but it can also overfit. By setting max_depth to None, the trees are allowed to grow until they contain fewer samples than the min_samples_split parameter or the data is completely homogeneous. This can allow the model to capture intricate patterns.

3. **n_jobs (Number of Cores Used for Parallelism):**
   - Value: -1 (Utilize all available cores)
   - Rationale: RandomForestClassifier can perform parallel computation during training, which speeds up the training process, especially for larger datasets. Setting n_jobs to -1 enables the model to use all available CPU cores, leading to faster training times.

4. **Other Hyperparameters:**
   - Other hyperparameters such as criterion (Gini impurity or entropy), min_samples_split (minimum number of samples required to split an internal node), and min_samples_leaf (minimum number of samples required to be at a leaf node) are not explicitly shown in the code but can significantly impact model performance.

The selection of these hyperparameters is based on a combination of experience, experimentation, and domain knowledge. The aim is to strike a balance between model complexity and generalization. The chosen values aim to create a RandomForestClassifier that can capture both simple and complex relationships in the data while avoiding overfitting. It's important to note that hyperparameter tuning is an iterative process that involves experimentation, evaluating model performance, and adjusting the parameters accordingly.
