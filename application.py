#!/usr/bin/python3
from flask import Flask, render_template, flash, request
# from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
import sys
import json
import pickle

app = Flask(__name__)
app.config.update(
    TESTING=True,
    SECRET_KEY=b'_5#y2L"F4Q8z\n\xec]/'
)

with open("svm.pickle", 'rb') as f:
    model = pickle.load(f)
classifier = model['classifier']
vectorizer = model['vectorizer']
mlb = model['mlb']
 
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/query', methods=['POST'])
def search():
    query = request.form.get('query')
    # print("function"+query)
    if not query:
        message = 'Please enter a search query!'
        print(message)
        return render_template('index.html', message=message)
    else:
        X = vectorizer.transform([query])
        print("Question: {}".format(query))
        prediction = classifier.predict(X)
        labels = mlb.inverse_transform(prediction)[0]
        # labels = ', '.join(labels)
        # if labels:
        #     message = "Predicted labels: {}".format(labels)
        # else:
        #     message = "No label available for this question"
        return render_template('index.html', message=labels)    
        # return render_template('index.html')    
        # return render_template('search_results.html',query=query,results=search_results['items'])    
   
 
if __name__ == "__main__":
    app.run(debug=True)