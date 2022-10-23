# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Ref: https://github.com/tidhamecha2/mlops-22/tree/feature/refactor
#PART: library dependencies -- sklear, torch, tensorflow, numpy, transformers

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.metrics import accuracy_score
import numpy as np

# Import helper functions
from utils import preprocess_digits, train_dev_test_split, param_tuning, train_save_model
from joblib import dump, load

# Class exercise:
# 1. Set the ranges of hyper parameters
gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]

train_frac = 0.8
test_frac = 0.1
dev_frac = 0.1

model_path = None

#PART: load dataset -- data from csv, tsv, jsonl, pickle
digits = datasets.load_digits()
data, image_data = preprocess_digits(digits)
X_train, y_train, X_dev, y_dev, X_test, y_test = train_dev_test_split(data, image_data, train_frac, dev_frac)

# Print the size of the images
print("\nSize of the images in the original dataset is: ", digits.images.shape[1],"x",digits.images.shape[2])

model_path, best_param_config, best_gamma, best_c = train_save_model(X_train, y_train, X_dev, y_dev, X_test, y_test, gamma_list, c_list, model_path)

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

print("\n", best_gamma, "," ,best_c, "\t", round(acc_train, 3), "\t", round(acc_dev, 3), "\t", round(acc_test, 3), "Best Combination\n")
print( np.unique(predicted_test))
