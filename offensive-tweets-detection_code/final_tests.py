import string
import pandas
import numpy
import nltk
import time
import math
import csv
from nltk.tokenize.casual import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score


# Read input file and return pandas dataframe with 'content' and 'label' columns
def read_file(input_file):
	# Load json data as records and remove columns where all values are NaN
	data = pandas.read_json(input_file, lines=True, orient="records").dropna(axis=1, how='all')
	return data


# Receives a dataframe with "content" as column and pre-processes the words
def pre_process_data(data, language, remove_punctuation=True, remove_stop_words=True, use_stemmer=False, lemmatization=True, punctuation=[], tokenizer=None, stemmer=None, lemmatizer=None):
	# Convert into lower case
	data['content'] = data['content'].apply(lambda x: x.lower())

	# Tokenize the tweets
	data['content'] = data['content'].apply(lambda x: tokenizer.tokenize(x))

	# Remove punctuation
	if remove_punctuation:
		data['content'] = data['content'].apply(lambda word_list: list(filter(lambda word: word not in punctuation, word_list)))

	# Remove stop words
	if remove_stop_words:
		# Download list of stop words
		nltk.download('stopwords')
		data['content'] = data['content'].apply(lambda word_list: list(filter(lambda word: word not in stopwords.words(language), word_list)))

	# Apply stemmer
	if use_stemmer:
		data['content'] = data['content'].apply(lambda word_list: [stemmer.stem(word) for word in word_list])

	# Apply lemmatization
	if lemmatization:
		# Download wordnet lemmatizer
		nltk.download('wordnet')
		data['content'] = data['content'].apply(lambda word_list: [lemmatizer.lemmatize(word) for word in word_list])

		return data


# Extract features from the data
def extract_features(data, max_features=25000, count_vectorizer=True, tf_idf=True, min_gram=1, max_gram=2):

	# N-gram (unigram + bigram)
	if(count_vectorizer):
		ngram_vectorizer = CountVectorizer(tokenizer=lambda x: x, ngram_range=(min_gram, max_gram), lowercase=False, max_features=max_features)
		X = ngram_vectorizer.fit_transform(data)
		scaler = MinMaxScaler(feature_range=(0, 1))
		features_X = [array for array in X.toarray()]
		features_X = scaler.fit_transform(features_X).tolist()

	# TF-IDF
	if(tf_idf):
		tf_idf = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False, max_features=max_features, norm='l2')
		X_2 = tf_idf.fit_transform(data)
		features_X_2 = [array for array in X_2.toarray()]

	# Aggregate features
	if(count_vectorizer and tf_idf):
		features_concatenation = numpy.hstack([features_X, features_X_2])
		features = [array for array in features_concatenation]
	elif(count_vectorizer):
		features = features_X
	else:
		features = features_X_2

	return features

# Data classification and evaluation of the predictions
def classify_and_evaluate(file, train_data, cross_validation_data, train_label, cross_validation_label):
	
	print("Logisticregression")
	classification_start = time.time()
	lr_classifier = LogisticRegression(penalty="l2", C=5, solver="liblinear").fit(train_data, train_label)
	classification_end = time.time()
	predictions = lr_classifier.predict(cross_validation_data)
	write_to_file(file, ["Logisticregression"] + [str(classification_end-classification_start)] + classification_results(cross_validation_label, predictions))

	print("LinearSVC")
	classification_start = time.time()
	lin_svc = LinearSVC(penalty="l2", C=5, loss="squared_hinge").fit(train_data, train_label)
	classification_end = time.time()
	predictions = lin_svc.predict(cross_validation_data)
	write_to_file(file, ["LinearSVC"] + [str(classification_end-classification_start)] + classification_results(cross_validation_label, predictions))

	print("DecisionTreeClassifier")
	classification_start = time.time()
	tree_classifier = DecisionTreeClassifier(criterion="gini", max_depth=600).fit(train_data, train_label)
	classification_end = time.time()
	predictions = tree_classifier.predict(cross_validation_data)
	write_to_file(file, ["DecisionTreeClassifier"] + [str(classification_end-classification_start)] + classification_results(cross_validation_label, predictions))

	print("MLPClassifier")
	classification_start = time.time()
	neural_net_classifier = MLPClassifier(hidden_layer_sizes=(20, 20), activation="relu", solver="adam", alpha=0.001, max_iter=500, early_stopping=True, batch_size="auto").fit(train_data, train_label)
	classification_end = time.time()
	predictions = neural_net_classifier.predict(cross_validation_data)
	write_to_file(file, ["MLPClassifier"] + [str(classification_end-classification_start)] + classification_results(cross_validation_label, predictions))


# Calculate the confusion matrix
def classification_results(cross_validation_label, predictions):
	tn, fp, fn, tp = confusion_matrix(cross_validation_label, predictions).ravel()
	results = [tp, tn, fp, fn, accuracy_score(cross_validation_label, predictions),
	precision_score(cross_validation_label, predictions),
	recall_score(cross_validation_label, predictions),
	f1_score(cross_validation_label, predictions)]
	return results

# Appends a row to file in csv format
def write_to_file(csv_file, row):
	writer = csv.writer(csv_file, delimiter=';')
	writer.writerow(row)


output_file = "final_output.csv"
language = "english"
remove_stop_words = True
remove_punctuation = True
use_stemmer = False
lemmatization = True
shuffle_data = True
punctuation = list(string.punctuation)
tokenizer = TweetTokenizer()
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
max_features = 25000
count_vectorizer = True
tf_idf = True
min_gram = 2
max_gram = 2

input_test_file = "test_data.json"
input_train_file = "train_data.json"

# Read file
train_data = read_file(input_train_file)
test_data = read_file(input_test_file)

train_data = pre_process_data(train_data, language, remove_punctuation, remove_stop_words, use_stemmer, lemmatization, punctuation, tokenizer, stemmer, lemmatizer)
test_data = pre_process_data(test_data, language, remove_punctuation, remove_stop_words, use_stemmer, lemmatization, punctuation, tokenizer, stemmer, lemmatizer)

all_features = pandas.concat([train_data, test_data])['content'].values
features = extract_features(all_features, max_features, count_vectorizer, tf_idf, min_gram, max_gram)
train_data['content'] = features[:len(train_data)]
test_data['content'] = features[len(train_data):]

test_data, train_data, test_label, train_label = test_data['content'].values, train_data['content'].values, test_data['label'].values, train_data['label'].values

train_data = numpy.vstack(train_data)
test_data = numpy.vstack(test_data)

file = open(output_file, "a+")
write_to_file(file, ["Classifier name", "Time to train", "True positives", "True negatives", "False positives", "False negatives", "Accuracy", "Precision", "Recall", "F1-Score"])

classify_and_evaluate(file, train_data, test_data, train_label, test_label)
