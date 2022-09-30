# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Ref: https://github.com/tidhamecha2/mlops-22/tree/feature/refactor
#PART: library dependencies -- sklear, torch, tensorflow, numpy, transformers

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics

# Import helper functions
from utils import preprocess_digits, train_dev_test_split, param_tuning

# Class exercise:
# 1. Set the ranges of hyper parameters
gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]

train_frac = 0.8
test_frac = 0.1
dev_frac = 0.1

#PART: load dataset -- data from csv, tsv, jsonl, pickle
digits = datasets.load_digits()

# Print the size of the images
print("\nSize of the images in the original dataset is: ", digits.images.shape[1],"x",digits.images.shape[2])

data, image_data = preprocess_digits(digits)
X_train, y_train, X_dev, y_dev, X_test, y_test = train_dev_test_split(data, image_data, train_frac, dev_frac)

# 2. Train for every combination of hyperparameter values
# 2a. Train the model
# 2b. Compute the acuracy on the validation set
# 3. Identify the best combination of hyperparameters for which validation set performance is maximum
# 4. Report the test set accuracy with this best model

# Variable for storing the accuracy of the current combination
acc = 0

#Variable for storing the current max accuracy
max_acc = 0;

# Variables for storing the current best hyper_parameter combination
best_gamma = 0
best_c = 0

# Variables for the different accuracies
predicted_train = 0
predicted_dev = 0
predicted_test = 0

print("\nGamma, C\t", "Train\t", "Dev\t", "Test\t")

#PART: Define the model
# Create a classifier: a support vector classifier
clf = svm.SVC()

max_acc, best_gamma, best_c = param_tuning(clf, gamma_list, c_list, X_train, y_train, X_dev, y_dev, X_test, y_test)

#PART: Get test set predictions
# Predict the value of the digit on the test subset

#PART: setting up hyperparameter
hyper_params = {'gamma':best_gamma, 'C':best_c}
clf.set_params(**hyper_params)

predicted_train = clf.predict(X_train)
predicted_dev = clf.predict(X_dev)
predicted_test = clf.predict(X_test)

acc_train = metrics.accuracy_score(y_pred=predicted_train, y_true=y_train)
acc_dev = metrics.accuracy_score(y_pred=predicted_dev, y_true=y_dev)
acc_test = metrics.accuracy_score(y_pred=predicted_test, y_true=y_test)

print("\n", best_gamma, "," ,best_c, "\t", round(acc_train, 3), "\t", round(acc_dev, 3), "\t", round(acc_test, 3), "Best Combination\n")
