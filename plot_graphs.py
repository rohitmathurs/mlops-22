# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Ref: https://github.com/tidhamecha2/mlops-22/tree/feature/refactor
#PART: library dependencies -- sklear, torch, tensorflow, numpy, transformers

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics, tree

# Import helper functions
from utils import preprocess_digits, train_dev_test_split, param_tuning, train_save_model
from joblib import dump, load

# Class exercise:
# 1. Set the ranges of hyper parameters for SVM
gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]

# Set the range of hyper parameters for decision tree classifier
max_depth_list = [3, 4, 5, 6]
max_leaf_nodes = [20, 30, 40, 50]

train_frac = 0.8
test_frac = 0.1
dev_frac = 0.1

model_path = None

#PART: load dataset -- data from csv, tsv, jsonl, pickle
digits = datasets.load_digits()

# Print the size of the images
print("\nSize of the images in the original dataset is: ", digits.images.shape[1],"x",digits.images.shape[2])

data, image_data = preprocess_digits(digits)

# Cleanup
del digits 

n_cv = 5

for i in range(1, n_cv+1):

	X_train, y_train, X_dev, y_dev, X_test, y_test = train_dev_test_split(data, image_data, train_frac, dev_frac)
	
	models_used = {"svm": svm.SVC(), "decision_tree": tree.DecisionTreeClassifier()}
	
	for clf_name in models_used:
		clf = models_used[clf_name]
	
		model_path, best_param_config, best_gamma, best_c = train_save_model(X_train, y_train, X_dev, y_dev, X_test, y_test, gamma_list, c_list, max_depth_list, max_leaf_nodes, model_path, clf)

		# Load the best model
		best_model = load(model_path)

		#PART: Get test set predictions
		# Predict the value of the digit on the test subset

		# PART: setting up hyperparameter
		# hyper_params = {'gamma':best_gamma, 'C':best_c}
		# clf.set_params(**hyper_params)

		predicted_train = best_model.predict(X_train)
		predicted_dev = best_model.predict(X_dev)
		predicted_test = best_model.predict(X_test)

		acc_train = metrics.accuracy_score(y_pred=predicted_train, y_true=y_train)
		acc_dev = metrics.accuracy_score(y_pred=predicted_dev, y_true=y_dev)
		acc_test = metrics.accuracy_score(y_pred=predicted_test, y_true=y_test)

		print("\nBest accuracy for Iteration #", i, "for model:", clf_name, "is: ", acc_dev)
