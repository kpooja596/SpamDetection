# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 10:50:50 2018

@author: Priyank Jain
"""
import pickle
import pandas as pd
import json
with open("spam_ham.json")as f:
    data=json.load(f)
df = pd.DataFrame.from_dict(data, orient='columns')

df=df.drop(df.columns[[0, 1, 2]], axis=1)

df_spam =pd.DataFrame(df["v2"][df["v1"]=="spam"])
df_ham=pd.DataFrame(df["v2"][df["v1"]=="ham"])

x=[]
for i in df_spam["v2"]:
    x.append(len(i))
df_spam["size"]=x
x=[]
for i in df_ham["v2"]:
    x.append(len(i))
df_ham["size"]=x
y=[]
for i in df["v2"]:
    y.append(len(i))
df["size"]=y

features=df.iloc[:,-1].values.reshape(-1,1)
labels=df.iloc[:,0].values

from sklearn.model_selection import train_test_split 
features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size=0.2,random_state=0) 

from sklearn.neighbors import KNeighborsClassifier 
classifier = KNeighborsClassifier(n_neighbors=5, p=2) 
classifier.fit(features_train, labels_train)

pred = classifier.predict(features_test) 


# Making the Confusion Matrix 
from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(labels_test, pred) 

score=classifier.score(features_train,labels_train)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = features_train, y = labels_train, cv = 10)
print ("mean accuracy is",accuracies.mean())
print (accuracies.std())
length_accuracies=accuracies.mean()

pickle_out=open("obj.pickle","wb")
pickle.dump(classifier,pickle_out)
pickle_out.close()
