# -*- coding: utf-8 -*-
"""
Created on Sun Dec 09 12:45:37 2018

@author: Priyank Jain
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import string
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle

import json
with open("spam_ham.json")as f:
    data=json.load(f)
data = pd.DataFrame.from_dict(data, orient='columns')

data=data.drop(data.columns[[0, 1, 2]], axis=1)

data = data.rename(columns={"v1":"class", "v2":"text"})
data.head()
data['length'] = data['text'].apply(len)

def pre_process(text):
   
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    words = ""
    for i in text:
            stemmer = SnowballStemmer("english")
            words += (stemmer.stem(i))+" "
    return words

textFeatures = data['text'].copy()

vectorizer = TfidfVectorizer("english")
features = vectorizer.fit_transform(textFeatures)

pickle_out=open("project_vector.pickle","wb")
pickle.dump(vectorizer,pickle_out)
pickle_out.close()

features_train, features_test, labels_train, labels_test = train_test_split(features, data['class'], test_size=0.3, random_state=111)


from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

svc = SVC(kernel='sigmoid', gamma=1.0)
svc.fit(features_train, labels_train)
prediction = svc.predict(features_test)
accuracy_score(labels_test,prediction)



from sklearn.naive_bayes import MultinomialNB

mnb = MultinomialNB(alpha=0.2)
mnb.fit(features_train, labels_train)
prediction = mnb.predict(features_test)
accuracy_score(labels_test,prediction)

pickle_out=open("project_mnb.pickle","wb")
pickle.dump(mnb,pickle_out)
pickle_out.close()


tex=raw_input("Enter msg: ")
tex=pre_process(tex)
tex1=[]
tex1.append(tex)
tex2=vectorizer.transform(tex1).toarray()
tex_predict=mnb.predict(tex2)
print(tex_predict[0])


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = mnb, X = features_train, y = labels_train, cv = 10)
print ("mean accuracy is",accuracies.mean())
print (accuracies.std())
nltk_accuracies=accuracies.mean()

list1=[]
list1.append(word_count_accuracies)
list1.append(length_accuracies)
list1.append(nltk_accuracies)
      
import numpy as np
import matplotlib.pyplot as plt
colour=["r","g","b"]
plt.xticks(np.arange(0,1,0.1))
plt.barh(["word_count","by_length","NLTK"],list1,color=colour)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test, prediction)