import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn import svm

data=pd.read_csv('Parkinsson disease.csv')
#print(data.head(),'\n',data.shape)

#print(data.isnull().sum())

#print(data.groupby('status').mean())

#splitinf the data
x=data.drop(columns=['name','status'])
y=data['status']
print(x)
print(y)

#stander scaller
scale=StandardScaler()
x=scale.fit_transform(x)

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.15,random_state=2)

#print(xtrain.shape,xtest.shape)

model=svm.SVC(kernel='linear')
model=model.fit(xtrain,ytrain)


#train acuraccy
trainprediction=model.predict(xtrain)
trainaccuracy=accuracy_score(ytrain,trainprediction)
print("train accuracy will be :",trainaccuracy)

#test acuraccy
testprediction=model.predict(xtest)
testaccuracy=accuracy_score(ytest,testprediction)
print("test accuracy will be :",testaccuracy)
