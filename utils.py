# Import datasets, classifiers and performance metrics
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Dataset preprocessing function
def preprocess_digits(dataset):
	#PART: data pre-processing -- to remove some noise, to normalize data, format the data to be consumed by mode
	# flatten the images
	n_samples = len(dataset.images)
	data = dataset.images.reshape((n_samples, -1))
	image_data = dataset.target
	return data, image_data

# Define split function
def train_dev_test_split(data, label, train_frac, dev_frac):
	#PART: define train/dev/test splits of experiment protocol
	# train to train model
	# dev to set hyperparameters of the model
	# test to evaluate the performance of the model
	dev_test_frac = 1-train_frac
	X_train, X_dev_test, y_train, y_dev_test = train_test_split(
	    data, label, test_size=dev_test_frac, shuffle=True
	)
	X_test, X_dev, y_test, y_dev = train_test_split(
	    X_dev_test, y_dev_test, test_size=(dev_frac)/dev_test_frac, shuffle=True
	)
	return X_train, y_train, X_dev, y_dev, X_test, y_test
	
# Parameter Tuning Function
def param_tuning(clf, gamma_list, c_list, X_train, y_train, X_dev, y_dev, X_test, y_test):
	#Variable for storing the current max accuracy
	max_acc = 0;

	# Variables for storing the current best hyper_parameter combination
	best_gamma = 0
	best_c = 0
	for gamma in gamma_list:
		for C in c_list:				
			#PART: setting up hyperparameter
			hyper_params = {'gamma':gamma, 'C':C}
			clf.set_params(**hyper_params)

			#PART: Train model
			# Learn the digits on the train subset
			clf.fit(X_train, y_train)

			#PART: Get dev set predictions
			# Predict the value of the digit on the test subset
			predicted_train = clf.predict(X_train)
			predicted_dev = clf.predict(X_dev)
			predicted_test = clf.predict(X_test)

			acc_train = metrics.accuracy_score(y_pred=predicted_train, y_true=y_train)
			acc_dev = metrics.accuracy_score(y_pred=predicted_dev, y_true=y_dev)
			acc_test = metrics.accuracy_score(y_pred=predicted_test, y_true=y_test)

			print(gamma, "," ,C, "\t", round(acc_train, 3), "\t", round(acc_dev, 3), "\t", round(acc_test, 3))

			if(acc_dev > max_acc):
				max_acc = acc_dev
				best_gamma = gamma
				best_c = C
	return max_acc, best_gamma, best_c
