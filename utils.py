# Import datasets, classifiers and performance metrics
from sklearn.model_selection import train_test_split
from sklearn import datasets, svm, metrics, tree
from joblib import dump, load


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
	    X_dev_test, y_dev_test, test_size=dev_frac / dev_test_frac, shuffle=True
	)
	return X_train, y_train, X_dev, y_dev, X_test, y_test
	
# Parameter Tuning Function
def param_tuning(clf, hyp_list_1, hyp_list_2, X_train, y_train, X_dev, y_dev, X_test, y_test):
	#Variable for storing the current max accuracy
	max_acc = -1
	best_gamma = 0
	best_c = 0
	best_max_depth = 0
	best_max_leaf_nodes = 0
	
	if type(clf) == svm.SVC:
		for param_1 in hyp_list_1:
			for param_2 in hyp_list_2:				
				#PART: setting up hyperparameter
				hyper_params = {'gamma':param_1, 'C':param_2}
				clf.set_params(**hyper_params)

				#PART: Train model
				# Learn the digits on the train subset
				clf.fit(X_train, y_train)

				#PART: Get dev set predictions
				# Predict the value of the digit on the test subset
				predicted_dev = clf.predict(X_dev)

				acc_dev = metrics.accuracy_score(y_pred=predicted_dev, y_true=y_dev)

				#print(gamma, "," ,C, "\t", round(acc_train, 3), "\t", round(acc_dev, 3), "\t", round(acc_test, 3))

				if(acc_dev > max_acc):
					max_acc = acc_dev
					best_gamma = param_1
					best_c = param_2
		# Now get the predictions on test set with the best parameters
		hyper_params = {'gamma':param_1, 'C':param_2}
		clf.set_params(**hyper_params)

		#PART: Train model
		# Learn the digits on the train subset
		clf.fit(X_train, y_train)

		#PART: Get dev set predictions
		# Predict the value of the digit on the test subset
		predicted_test = clf.predict(X_test)

		max_acc = metrics.accuracy_score(y_pred=predicted_test, y_true=y_test)
		return 	max_acc
	else:
		for param_1 in hyp_list_1:
			for param_2 in hyp_list_2:				
				#PART: setting up hyperparameter
				hyper_params = {'max_depth':param_1, 'max_leaf_nodes':param_2}
				clf.set_params(**hyper_params)

				#PART: Train model
				# Learn the digits on the train subset
				clf.fit(X_train, y_train)

				#PART: Get dev set predictions
				predicted_dev = clf.predict(X_dev)

				acc_dev = metrics.accuracy_score(y_pred=predicted_dev, y_true=y_dev)

				if(acc_dev > max_acc):
					max_acc = acc_dev
					best_max_depth = param_1
					best_max_leaf_nodes = param_2
					
		# Now get the predictions on test set with the best parameters
		hyper_params = {'max_depth':param_1, 'max_leaf_nodes':param_2}
		clf.set_params(**hyper_params)

		#PART: Train model
		# Learn the digits on the train subset
		clf.fit(X_train, y_train)

		#PART: Get dev set predictions
		# Predict the value of the digit on the test subset
		predicted_test = clf.predict(X_test)

		max_acc = metrics.accuracy_score(y_pred=predicted_test, y_true=y_test)
		return 	max_acc
	
