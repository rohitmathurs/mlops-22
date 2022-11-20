import sys
import os
sys.path.append(".")
from utils import preprocess_digits, train_dev_test_split
from sklearn.model_selection import train_test_split
from sklearn import datasets

def test_same_random_seed():
	digits = datasets.load_digits()
	data, image_data = preprocess_digits(digits)
	sample_data = data[:500]
	sample_image_data = image_data[:500]
	train_frac = 0.8
	dev_frac = 0.1
	dev_test_frac = 1-train_frac
	seed1 = 20112022
	seed2 = 20112022
	
	X_train, y_train, X_dev, y_dev, X_test, y_test = train_dev_test_split(sample_data, sample_image_data, train_frac, dev_frac, seed1)
	X_train1, y_train1, X_dev1, y_dev1, X_test1, y_test1 = train_dev_test_split(sample_data, sample_image_data, train_frac, dev_frac, seed2)
	
	assert X_train.all() == X_train1.all()
	assert y_train.all() == y_train1.all()
	assert X_dev.all() == X_dev1.all()
	assert y_dev.all() == y_dev1.all()
	assert X_test.all() == X_test1.all()
	assert y_test.all() == y_test1.all()
	
def test_different_random_seed():
	digits = datasets.load_digits()
	data, image_data = preprocess_digits(digits)
	sample_data = data[:500]
	sample_image_data = image_data[:500]
	train_frac = 0.8
	dev_frac = 0.1
	dev_test_frac = 1-train_frac
	seed1 = 20112022
	seed2 = 20
	
	X_train, y_train, X_dev, y_dev, X_test, y_test = train_dev_test_split(sample_data, sample_image_data, train_frac, dev_frac, seed1)
	X_train1, y_train1, X_dev1, y_dev1, X_test1, y_test1 = train_dev_test_split(sample_data, sample_image_data, train_frac, dev_frac, seed2)
	
	assert X_train.all() == X_train1.all()
	assert y_train.all() == y_train1.all()
	assert X_dev.all() == X_dev1.all()
	assert y_dev.all() == y_dev1.all()
	assert X_test.all() == X_test1.all()
	assert y_test.all() == y_test1.all()
