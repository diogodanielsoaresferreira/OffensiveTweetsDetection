from cybertrolls_analysis_dt import perform_analysis

max_features = 25000

output_file = "output_decision_tree.csv"
count_vectorizer = True
tf_idf = True
min_gram = 2
max_gram = 2
parameters = []

max_depth = [100, 200, 400, 600, 800]
criterions = ["gini", "entropy"]

for criterion in criterions:
	for depth in max_depth:
		parameters.append({"criterion": criterion, "depth": depth})

perform_analysis(output_file=output_file, count_vectorizer=count_vectorizer, tf_idf=tf_idf, min_gram=min_gram, max_gram=max_gram, max_features=max_features, parameters=parameters)
