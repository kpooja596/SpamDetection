# -*- coding: utf-8 -*-
"""
Created on Sun Dec 09 14:05:17 2018

@author: Priyank Jain
"""

from flask import Flask, request, render_template
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('index.html')

@app.route('/approach', methods=['POST'])
def my_form_post():
    value=request.form['gets']
    text = request.form['comment']
    if value=='nltk':
        
        def preprocess(text):
            text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
            words = ""
            for i in text:
                    stemmer = SnowballStemmer("english")
                    words += (stemmer.stem(i))+" "
            return words
        text=preprocess(text)
        pri=[]
        pri.append(text)
        pickle_in_vectorizer=open("project_vector.pickle","rb")
        vectorizer=pickle.load(pickle_in_vectorizer)
        text=vectorizer.transform(pri).toarray()
        pickle_in_mnb=open("project_mnb.pickle","rb")
        classifier_mnb=pickle.load(pickle_in_mnb)
        labels_predict = classifier_mnb.predict(text)
        processed_text = labels_predict[0].upper()
        print processed_text
        return render_template("result.html",result=processed_text)
    elif value=="length":
        processed_text=len(text)
        pickle_in_len=open("obj.pickle","rb")
        classifier_len=pickle.load(pickle_in_len)
        pred=classifier_len.predict(processed_text)
        pred=pred[0].upper()
        print pred
        return render_template("result.html",result=pred)
    elif value=="word":
         text=text.strip()
         count=len(text.split())
         pickle_in_wrd=open("wordcount.pickle","rb")
         classifier_wrd=pickle.load(pickle_in_wrd)
         pred=classifier_wrd.predict(count)
         print count
         pred=pred[0].upper()
         print pred
         return render_template("result.html",result=pred)
         

app.run()