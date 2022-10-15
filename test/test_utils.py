import sys
import os
sys.path.append(".")
from utils import preprocess_digits, train_save_model
from sklearn import datasets
# import pdb

# Some test cases that will validate if models are indeed getting saved or not

# Step 1: Train on a small dataset and provide a path to save trained model
# Step 2: Assert if the file exists at the provided path
# Step 3: Assert if the file is indeed a sckit-learn model
# Step 4: Optionally check the checksum or MD5 of the model file

def test_model_saved():
	model_path = None
	digits = datasets.load_digits()
	data, image_data = preprocess_digits(digits)
	sample_data = data[:500]
	sample_image_data = image_data[:500]
	gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
	c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]
#	pdb.set_trace()
	actual_model_path, best_gamma, best_c, clf = train_save_model(sample_data, sample_image_data, sample_data, sample_image_data, sample_data, sample_image_data, gamma_list, c_list, model_path)
	assert actual_model_path == model_path 
	
	assert os.path.exists(model_path)
	loaded_model = load(model_path)
	assert type(loaded_model) == type(clf)
