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
def classify_and_evaluate(file, train_data, cross_validation_data, train_label, cross_validation_label, number_features, parameters):

	classification_start = time.time()
	decision_tree_classifier = MLPClassifier(hidden_layer_sizes=parameters["layers"], activation="relu", solver="adam", alpha=parameters["lambda"], max_iter=500, early_stopping=True, batch_size="auto").fit(train_data, train_label)
	classification_end = time.time()
	predictions = decision_tree_classifier.predict(cross_validation_data)
	write_to_file(file, ["MLPClassifier"] + [str(number_features), str(classification_end - classification_start)] + classification_results(cross_validation_label, predictions))
	

# Calculate the confusion matrix
def classification_results(cross_validation_label, predictions):
	tn, fp, fn, tp = confusion_matrix(cross_validation_label, predictions).ravel()
	return [tp, tn, fp, fn, accuracy_score(cross_validation_label, predictions),
	precision_score(cross_validation_label, predictions),
	recall_score(cross_validation_label, predictions),
	f1_score(cross_validation_label, predictions)]


# Appends a row to file in csv format
def write_to_file(csv_file, row):
	writer = csv.writer(csv_file, delimiter=';')
	writer.writerow(row)


def perform_analysis(
		input_file = "train_data.json",
		output_file = "output.csv",
		language = "english",
		number_folds = 5,
		remove_stop_words = True,
		remove_punctuation = True,
		use_stemmer = False,
		lemmatization = True,
		shuffle_data = True,
		punctuation = list(string.punctuation),
		tokenizer = TweetTokenizer(),
		stemmer = PorterStemmer(),
		lemmatizer = WordNetLemmatizer(),
		max_features = 1000,
		count_vectorizer = True,
		tf_idf = True,
		min_gram = 1,
		max_gram = 2,
		parameters = {}
	):

	# Read file
	read_start = time.time()
	data = read_file(input_file)
	read_end = time.time()

	print("Pre-processing...")
	# Data pre-processing
	processing_start = time.time()
	data = pre_process_data(data, language, remove_punctuation, remove_stop_words, use_stemmer, lemmatization, punctuation, tokenizer, stemmer, lemmatizer)
	processing_end = time.time()

	print("Feature extraction...")
	# Feature extraction
	extraction_start = time.time()
	data['content'] = extract_features(data['content'].values, max_features, count_vectorizer, tf_idf, min_gram, max_gram)
	extraction_end = time.time()

	# Number of features
	number_features = len(data['content'][0])

	for parameters_dict in parameters:
		# Apply 5-fold cross-validation
		# Split on train and cross-validation data
		kf = model_selection.KFold(n_splits=number_folds, shuffle=shuffle_data)

		file = open(output_file, "a+")
		for key, value in parameters_dict.items():
			write_to_file(file, [key, value])
		write_to_file(file, ["Classifier name", "Number of features", "Time to train", "True positives", "True negatives", "False positives", "False negatives", "Accuracy", "Precision", "Recall", "F1-Score"])

		# Classification
		classify_start = time.time()
		iteration = 1
		for train_index, cross_validation_index in kf.split(data['content'].values):
			print("Classification - iteration {}...".format(iteration))

			train_data, cross_validation_data, train_label, cross_validation_label = data['content'].values[train_index], data['content'].values[cross_validation_index], data['label'].values[train_index], data['label'].values[cross_validation_index]
			train_data = numpy.vstack(train_data)
			cross_validation_data = numpy.vstack(cross_validation_data)

			classify_and_evaluate(file, train_data, cross_validation_data, train_label, cross_validation_label, number_features, parameters_dict)

			iteration += 1

			classify_end = time.time()

		# Time results
		print("Total time: {:.2f} seconds".format(classify_end - read_start))
		print("- Read time: {:.2f} seconds".format(read_end - read_start))
		print("- Data pre-processing time: {:.2f} seconds".format(processing_end - processing_start))
		print("- Feature extraction time: {:.2f} seconds".format(extraction_end - extraction_start))
		print("- Total classification time: {:.2f} seconds, {:.2f} per iteration".format(classify_end - classify_start, (classify_end - classify_start)/number_folds))


if __name__ == "__main__":
	perform_analysis()