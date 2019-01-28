from cybertrolls_analysis_lr_lsvc import perform_analysis

max_features = 25000
'''
output_file = "output_logistic_regression.csv"
count_vectorizer = True
tf_idf = True
min_gram = 2
max_gram = 2
classifier = 0
penalty_list = ["l1", "l2"]
c_list = [0.1, 1, 5, 10, 25, 50, 100, 500, 1000]
solver_list = ["liblinear", "saga"]
parameters = [{"penalty": penalty, "C": c, "solver": solver} for penalty in penalty_list for c in c_list for solver in solver_list]

perform_analysis(output_file=output_file, count_vectorizer=count_vectorizer, tf_idf=tf_idf, min_gram=min_gram, max_gram=max_gram, max_features=max_features, classifier=classifier, parameters=parameters)
'''
###

output_file = "output_linear_svc.csv"
count_vectorizer = True
tf_idf = True
min_gram = 2
max_gram = 2
classifier = 1
penalty_list = ["l1", "l2"]
c_list = [0.1, 1, 5, 10, 25, 50, 100, 500, 1000]
max_iter_list = [1000, 2000, 5000]
loss_list = ["squared_hinge", "hinge"]
parameters = [{"penalty": penalty, "C": c, "max_iter": max_iter, "loss": loss} for penalty in penalty_list for c in c_list for max_iter in max_iter_list for loss in loss_list]
perform_analysis(output_file=output_file, count_vectorizer=count_vectorizer, tf_idf=tf_idf, min_gram=min_gram, max_gram=max_gram, max_features=max_features, classifier=classifier, parameters=parameters)

