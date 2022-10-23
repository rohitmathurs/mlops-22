import sys
import os
sys.path.append(".")
import numpy as np
from utils import preprocess_digits, train_save_model, train_dev_test_split
from sklearn import datasets
from joblib import dump, load

# import pdb

# Some test cases that will validate if models are indeed getting saved or not

# Step 1: Train on a small dataset and provide a path to save trained model
# Step 2: Assert if the file exists at the provided path
# Step 3: Assert if the file is indeed a sckit-learn model
# Step 4: Optionally check the checksum or MD5 of the model file

def test_imbalanced_classifier():
	model_path = None
	train_frac = 0.8
	test_frac = 0.1
	dev_frac = 0.1
	digits = datasets.load_digits()
	data, image_data = preprocess_digits(digits)
	sample_data = data[:500]
	sample_image_data = image_data[:500]
	gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
	c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]
#	pdb.set_trace()
	X_train, y_train, X_dev, y_dev, X_test, y_test = train_dev_test_split(data, image_data, train_frac, dev_frac)
	actual_model_path, best_gamma, best_c, clf = train_save_model(sample_data, sample_image_data, sample_data, sample_image_data, sample_data, sample_image_data, gamma_list, c_list, model_path)
	
	best_model = load(actual_model_path)
	predicted_test = best_model.predict(X_test)
	
	classifiers = np.unique(predicted_test)
	true_classifiers = np.unique(image_data)
	assert len(classifiers) == len(true_classifiers)
