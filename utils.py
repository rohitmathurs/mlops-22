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
	    X_dev_test, y_dev_test, test_size=(dev_frac)/dev_test_frac, shuffle=True
	)
	return X_train, y_train, X_dev, y_dev, X_test, y_test
	
# Parameter Tuning Function
def param_tuning(clf, gamma_list, c_list, max_depth_list, max_leaf_nodes, X_train, y_train, X_dev, y_dev, X_test, y_test):
	#Variable for storing the current max accuracy
	max_acc = -1

	# Variables for storing the current best hyper_parameter combination
	best_gamma = 0
	best_c = 0
	best_max_depth_list = 0
	best_max_leaf_nodes = 0
	best_model = None
	
	if type(clf) == svm.SVC:
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

				#print(gamma, "," ,C, "\t", round(acc_train, 3), "\t", round(acc_dev, 3), "\t", round(acc_test, 3))

				if(acc_dev > max_acc):
					max_acc = acc_dev
					best_gamma = gamma
					best_c = C
					best_model = clf
		return max_acc, best_gamma, best_c, best_model, best_max_depth_list, best_max_leaf_nodes
	else:	
		for max_depth in max_depth_list:
			for leaf in max_leaf_nodes:				
				#PART: setting up hyperparameter
				hyper_params = {'max_depth':max_depth, 'max_leaf_nodes':leaf}
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

				#print(gamma, "," ,C, "\t", round(acc_train, 3), "\t", round(acc_dev, 3), "\t", round(acc_test, 3))

				if(acc_dev > max_acc):
					max_acc = acc_dev
					best_max_depth_list = max_depth
					best_max_leaf_nodes = leaf
					best_model = clf
		return max_acc, best_gamma, best_c, best_model, best_max_depth_list, best_max_leaf_nodes
	
def train_save_model(X_train, y_train, X_dev, y_dev, X_test, y_test, gamma_list, c_list, max_depth_list, max_leaf_nodes, model_path, clf):

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
	best_max_depth_list = 0
	best_max_leaf_nodes = 0
	best_model = None

	# Variables for the different accuracies
	predicted_train = 0
	predicted_dev = 0
	predicted_test = 0

	#print("\nGamma, C\t", "Train\t", "Dev\t", "Test\t")

	#PART: Define the model
	# Create a classifier: a support vector classifier
	#clf = svm.SVC()

	max_acc, best_gamma, best_c, best_model, best_max_depth_list, best_max_leaf_nodes = param_tuning(clf, gamma_list, c_list, max_depth_list, max_leaf_nodes, X_train, y_train, X_dev, y_dev, X_test, y_test)

	# Save the best_model
	best_param_config = "_".join(["Gamma=" + str(best_gamma) + "_C=" + str(best_c)])
	if model_path is None:
		model_path = "svm_" + best_param_config + ".joblib" 
	dump(best_model, model_path)
	
	return model_path, best_param_config, best_gamma, best_c
	
