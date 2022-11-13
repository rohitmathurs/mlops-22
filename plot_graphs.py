# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Ref: Class 
#PART: library dependencies -- sklear, torch, tensorflow, numpy, transformers

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics, tree
import numpy as np
import statistics as st

# Import helper functions
from utils import preprocess_digits, train_dev_test_split, param_tuning
from joblib import dump, load

# Class exercise:
# 1. Set the ranges of hyper parameters for SVM
gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]

# Set the ranges of hyper parameters for decision tree classifier
max_depth_list = [3, 5, 7, 9]
max_leaf_nodes = [30, 40, 50, 60]

# Set number of iterations (Choosing equal to the test train splits)
n_cv = 5

# Set the ranges for different ranges

train_frac = [0.6, 0.65, 0.7, 0.75, 0.8]
dev_frac = 0.1

model_path = None

# Create arrays to store the max accuracies for all runs. (Required for statistics)
max_acc_svm = np.zeros(n_cv)
max_acc_dtc = np.zeros(n_cv)

#PART: load dataset -- data from csv, tsv, jsonl, pickle
digits = datasets.load_digits()
data, image_data = preprocess_digits(digits)

# Cleanup
del digits 

# Header print of the screen output
print("\nRun\t", "SVM\t\t\t", "Decsion_Tree\t")

for i in range(n_cv):
	
	# Choose a different test train split on every iteration
	X_train, y_train, X_dev, y_dev, X_test, y_test = train_dev_test_split(data, image_data, train_frac[i], dev_frac)
	
	models_used = {"svm": svm.SVC(), "decision_tree": tree.DecisionTreeClassifier()}
	
	# Run the dataset through both models and get max accuracies based on the best tuned parameters
	for clf_name in models_used:
		clf = models_used[clf_name]
		if type(clf) ==svm.SVC:
			max_acc_svm[i] = param_tuning (clf, gamma_list, c_list, X_train, y_train, X_dev, y_dev, X_test, y_test)
		else:
			max_acc_dtc[i] = param_tuning (clf, max_depth_list, max_leaf_nodes, X_train, y_train, X_dev, y_dev, X_test, y_test)

	# Print the max accuracies found for each model
	print("\n", i+1, "\t", max_acc_svm[i], "\t", max_acc_dtc[i], "\n")
	
# Print the mean and standard deviations
print("\nMean\t", st.mean(max_acc_svm), "\t", st.mean(max_acc_dtc), "\n")
print("\nStdDev\t", st.stdev(max_acc_svm), "\t", st.stdev(max_acc_dtc), "\n")
