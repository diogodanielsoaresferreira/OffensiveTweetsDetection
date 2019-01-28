from cybertrolls_analysis import perform_analysis

max_features = 10000

output_file = "output_1-gram_10000.csv"
count_vectorizer = True
tf_idf = False
min_gram = 1
max_gram = 1
perform_analysis(output_file=output_file, count_vectorizer=count_vectorizer, tf_idf=tf_idf, min_gram=min_gram, max_gram=max_gram, max_features=max_features)

output_file = "output_2-gram_10000.csv"
count_vectorizer = True
tf_idf = False
min_gram = 2
max_gram = 2
perform_analysis(output_file=output_file, count_vectorizer=count_vectorizer, tf_idf=tf_idf, min_gram=min_gram, max_gram=max_gram, max_features=max_features)

output_file = "output_1-2-gram_10000.csv"
count_vectorizer = True
tf_idf = False
min_gram = 1
max_gram = 2
perform_analysis(output_file=output_file, count_vectorizer=count_vectorizer, tf_idf=tf_idf, min_gram=min_gram, max_gram=max_gram, max_features=max_features)

output_file = "output_tfidf_10000.csv"
count_vectorizer = False
tf_idf = True
min_gram = 1
max_gram = 2
perform_analysis(output_file=output_file, count_vectorizer=count_vectorizer, tf_idf=tf_idf, min_gram=min_gram, max_gram=max_gram, max_features=max_features)

output_file = "output_1-gram_tfidf_10000.csv"
count_vectorizer = True
tf_idf = True
min_gram = 1
max_gram = 1
perform_analysis(output_file=output_file, count_vectorizer=count_vectorizer, tf_idf=tf_idf, min_gram=min_gram, max_gram=max_gram, max_features=max_features)


output_file = "output_2-gram_tfidf_10000.csv"
count_vectorizer = True
tf_idf = True
min_gram = 2
max_gram = 2
perform_analysis(output_file=output_file, count_vectorizer=count_vectorizer, tf_idf=tf_idf, min_gram=min_gram, max_gram=max_gram, max_features=max_features)


output_file = "output_1-2-gram_tfidf_10000.csv"
count_vectorizer = True
tf_idf = True
min_gram = 1
max_gram = 2
perform_analysis(output_file=output_file, count_vectorizer=count_vectorizer, tf_idf=tf_idf, min_gram=min_gram, max_gram=max_gram, max_features=max_features)

