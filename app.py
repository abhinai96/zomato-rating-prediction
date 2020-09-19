import pickle
from flask import Flask,render_template,request
import numpy as np

model=pickle.load(open('zomato2.pkl','rb')) #loading the model
app=Flask(__name__)
@app.route('/')
def man():
    return render_template("index.html")
@app.route('/predict',methods=["POST"])
def home():
    int_features=[int(x) for x in request.form.values()]
    final_features=[np.array(int_features)]
    prediction=model.predict(final_features)
    output=round(prediction[0],1)
    return render_template ("index.html",prediction_text="Rating {}".format(output))
if __name__=="__main__":
    app.run(debug=True)