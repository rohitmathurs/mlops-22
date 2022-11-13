from flask import Flask
from flask import request
from joblib import load


app = Flask(__name__)

#Load one of the saved models
model_path = "../svm_Gamma=0.01_C=0.5.joblib"
model = load(model_path)

@app.route("/")
def hello_world():
    return "<p>Hello, MlOps Class!</p>"
    

@app.route("/sum", methods=['POST'])
def sum():
	x = request.json['x']
	y = request.json['y']
	z = x + y
	return {"sum":z}
	
@app.route("/predict", methods=['POST'])
def predict_digit():
	image = request.json['image']
	print("Loaded model")
	predicted = model.predict([image])
	return {"y_predicted":int(predicted[0])}
