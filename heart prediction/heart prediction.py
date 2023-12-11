import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data=pd.read_csv('heart_disease_data.csv')

#print(data.head())
#print(data.shape)
#print(data.describe())

#print(data.isnull().sum())

x=data.drop(columns=['target'])
y=data['target']

#print(x)
#print(y)

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=3)

model=LogisticRegression()
model.fit(xtrain,ytrain)

#train acuraccy
trainprediction=model.predict(xtrain)
trainaccuracy=accuracy_score(ytrain,trainprediction)
print("train accuracy will be :",trainaccuracy)

#test acuraccy
testprediction=model.predict(xtest)
testaccuracy=accuracy_score(ytest,testprediction)
print("test accuracy will be :",testaccuracy)

