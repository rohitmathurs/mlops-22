import sys
from sklearn import datasets
import pdb
from utils import preprocess_digits, train_save_model

# Some test cases that will validate if models are indeed getting saved or not

# Step 1: Train on a small dataset and provide a path to save trained model
# Step 2: Assert if the file exists at the provided path
# Step 3: Assert if the file is indeed a sckit-learn model
# Step 4: Optionally check the checksum or MD5 of the model file

digits = datasets.load_digits()
data, image_data = preprocess_digits(digits)
data = data[:500]
image_data = image_data[:500]
pdb.set_trace()

def test_model_saved():
	model_path = None
	train_save_model(X_train, y_train, X_dev, y_dev, X_test, y_test, gamma_list, c_list, model_path)
