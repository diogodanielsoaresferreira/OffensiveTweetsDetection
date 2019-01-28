from cybertrolls_analysis_nn import perform_analysis

max_features = 25000

output_file = "output_mlp.csv"
count_vectorizer = True
tf_idf = True
min_gram = 2
max_gram = 2
parameters = []

lambda_values = [0, 0.0001, 0.001, 0.01, 0.1, 1, 10]
hidden_layers = [(100,), (10,10,), (20,20,), (50,10,), (10,10,10,), (20,20,20,), (50,50,10,)]

for layers in hidden_layers:
	for lambda_value in lambda_values:
		parameters.append({"lambda": lambda_value, "layers": layers})

perform_analysis(output_file=output_file, count_vectorizer=count_vectorizer, tf_idf=tf_idf, min_gram=min_gram, max_gram=max_gram, max_features=max_features, parameters=parameters)
