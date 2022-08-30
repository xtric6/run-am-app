from flask import Flask,render_template,url_for,request
import joblib
import os

#vectorizer
news_vectorizer = open(os.path.join("static/vectorizer.pkl"),"rb")
news_cv = joblib.load(news_vectorizer)

app = Flask(__name__)

def get_keys(val,my_dict):
	for key,value in my_dict.items():
		if val == value:
			return key


@app.route('/')
def index():
	return render_template("index3.html")

@app.route('/predict', methods=['GET','POST'])
def predict():
	if request.method=='POST':
		rawtext = request.form['rawtext']
		vectorized_text = news_cv.transform([rawtext]).toarray()
		model = open(os.path.join("static/news_predictor_PAC.pkl"),"rb")
		news_clf = joblib.load(model)


	#prediction
	prediction_labels = {'This news is Verified':1,'This news is Unverified':0}
	prediction = news_clf.predict(vectorized_text)
	final_result = get_keys(prediction,prediction_labels)

	return render_template("index3.html",rawtext=rawtext.capitalize(),final_result=final_result)

if __name__== '__main__':
	app.run(debug=True)