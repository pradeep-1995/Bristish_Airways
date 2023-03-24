# Bristish_Airways
## Task-1   Web scraping review and analysis
This Jupyter notebook includes some code to get you started with web scraping. We will use a package called BeautifulSoup to collect the data from the web. Once you've collected your data and saved it into a local .csv file you should start with your analysis.

### 1. Scraping data from Skytrax

If you visit [https://www.airlinequality.com] you can see that there is a lot of data there. For this task, we are only interested in reviews related to British Airways and the Airline itself.

If you navigate to this link: [https://www.airlinequality.com/airline-reviews/british-airways] you will see this data. Now, we can use Python and BeautifulSoup to collect all the links to the reviews and then to collect the text data on each of the individual review links.

#### Rule Based Approch
This is a practical approach to analyzing text without training or using machine learning models. The result of this approach is a set of rules based on which the text is labeled as positive/negative/neutral. These rules are also known as lexicons. Hence, the Rule-based approach is called Lexicon based approach.

A lexicon refers to a collection of words or phrases with associated information such as definitions, parts of speech, semantic and syntactic information, and more. Lexicons can be used in various NLP tasks, including sentiment analysis, text classification, named entity recognition, and more.

Widely used lexicon-based approaches are TextBlob, VADER, SentiWordNet.

#### Data preprocessing steps:
Cleaning the text ,
Tokenization ,
Enrichment – POS tagging ,
Stopwords removal ,
Obtaining the stem words
### 2. Cleaning Text
### 3. Tokenization Text
Tokenization is the process of breaking down a text document into smaller units of text called tokens. These tokens could be words, phrases, or even individual characters. The main purpose of tokenization is to convert the text into a format that can be easily processed by a computer program. Tokenization is a common preprocessing step in natural language processing (NLP) tasks such as text classification, sentiment analysis, and machine translation.

Tokenization is the process of breaking the text into smaller pieces called Tokens. It can be performed at sentences(sentence tokenization) or word level(word tokenization).

This punkt tokenizer divides a text into a list of sentences by using an unsupervised algorithm to build a model for abbreviation words, collocations, and words that start sentences.
### 4. POS_tagging Text
Parts of Speech (POS) tagging is a process of converting each token into a tuple having the form (word, tag). POS tagging essential to preserve the context of the word and is essential for Lemmatization.

To perform POS tagging using Spacy, you first need to install Spacy and download a language model.
### 5. STOP_WORDS_Removal Text
Stopwords in English are words that carry very little useful information. We need to remove them as part of text preprocessing. nltk has a list of stopwords of every language.

Here we dont remove stopwords because it may help for sentimate analysis.
### 6. Obtaining the stem words ( Lemmatization )
A stem is a part of a word responsible for its lexical meaning. The two popular techniques of obtaining the root/stem words are Stemming and Lemmatization.

The key difference is Stemming often gives some meaningless root words as it simply chops off some characters in the end. Lemmatization gives meaningful root words, however, it requires POS tags of the words.

Lemmatization is the process of reducing words to their base or root form. In Natural Language Processing, it is used to group together different forms of a word so they can be analyzed as a single item.

NLTK provides an interface to the WordNet lemmatizer which can be used to perform lemmatization on text.
### 7. Sentiment Analysis using VADER
Sentiment Analysis is the process of identifying and extracting the sentiment behind a piece of text, i.e., whether the text expresses a positive, negative, or neutral sentiment. VADER (Valence Aware Dictionary and Sentiment Reasoner) is a pre-trained sentiment analysis model in Python that can analyze the sentiment of a piece of text and classify it as either positive, negative, or neutral.
### 8. Visualization
Review Analysis of Positive,Neagative and Neutral Review.
### 9. Word Cloud Visualization
Word Cloud or Tag Clouds is a visualization technique for texts that are natively used for visualizing the tags or keywords from the websites.
### 10. Export Cleaned Dataset
## Task-2 Predictive modeling of customer bookings
This Jupyter notebook includes some code to get you started with this predictive modeling task. We will use various packages for data manipulation, feature engineering and machine learning.

Exploratory data analysis
First, we must explore the data in order to better understand what we have and the statistical properties of the dataset.
### 1. Data Collection
Upload widget is only available when the cell has been executed in the current browser session. Please rerun this cell to enable.
Saving customer_booking.csv to customer_booking.csv
Chardet is a Python library that is used to automatically detect the encoding of text. It is short for "charset detection" and is commonly used to determine the character encoding of web pages and other text files.
num_passengers = number of passengers travelling.

sales_channel = sales channel booking was made on

trip_type = trip Type (Round Trip, One Way, Circle Trip)

purchase_lead = number of days between travel date and booking date

length_of_stay = number of days spent at destination

flight_hour = hour of flight departure

flight_day = day of week of flight departure

route = origin -> destination flight route

booking_origin = country from where booking was made

wants_extra_baggage = if the customer wanted extra baggage in the booking

wants_preferred_seat = if the customer wanted a preferred seat in the booking

wants_in_flight_meals = if the customer wanted in-flight meals in the booking

flight_duration = total duration of flight (in hours)

booking_complete = flag indicating if the customer completed the booking
### 2. Exploratory Data Analysis (EDA)
### 3. Visualization
### 4. Feature Importance
It is a function provided by the scikit-learn machine learning library in Python. It is used for feature selection in classification tasks, based on the concept of mutual information. Mutual information is a measure of the amount of information that one variable provides about another variable. In the context of feature selection, mutual information is used to estimate the dependence between a feature and the target variable.

mutual_info_classif is specifically designed for classification problems, where the target variable is categorical. It computes the mutual information between each feature and the target variable, and returns an array of scores, where each score corresponds to a feature.

The mutual_info_classif function can be useful in feature selection or feature engineering tasks, where we want to identify the most informative features that are relevant for predicting the target variable. By selecting the top-ranked features based on their mutual information score, we can reduce the dimensionality of the feature space and potentially improve the performance of a classification model.
### 5. Train_test_split
### 6. Feature Scaling
MinMax scaling is a type of data normalization that is commonly used in machine learning to transform numerical data into a range between 0 and 1. The process involves subtracting the minimum value in the data set from each data point and then dividing the result by the range of the data set (i.e., the difference between the maximum and minimum values).

The formula for MinMax scaling is as follows:

x_scaled = (x - min(x)) / (max(x) - min(x))
### 7. One-Hot-Encoding (ohe)
It is a technique used to represent categorical variables as numerical data in machine learning. In this technique, each category is represented as a binary vector, with one element of the vector set to 1 and all others set to 0.

### 8. Model_Selection
#### Model 1 : Random forest classifier with top 6 features
#### Model 2 : Random forest classifier with All features
#### Model 3 : XGB classifier with top 6 features
#### Model 4 : XGB classifier with all features

# Task 1 -[https://github.com/pradeep-1995/Bristish_Airways/blob/main/Task%202%20-%20Resource.pptx]
# Task 2 -[https://github.com/pradeep-1995/Bristish_Airways/blob/main/British_airways_task_2.ipynb]
