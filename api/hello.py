from flask import Flask
from flask import request
from joblib import load

app = Flask(__name__)

#Load one of the saved models
model_path = "../svm_Gamma=0.01_C=0.5.joblib"
model = load(model_path)
	
@app.route("/predict", methods=['POST'])
def predict_digit():
	image1 = request.json['image1']
	image2 = request.json['image2']
	print("Loaded model")
	
	predicted = model.predict([image1])
	image1_predicted = int(predicted[0])
	
	predicted = model.predict([image2])
	image2_predicted = int(predicted[0])

	if image1_predicted == image2_predicted:
		return '\nInput images are the same digit\n\n'
	else:
		return '\nInput images are not the same digit\n\n'
