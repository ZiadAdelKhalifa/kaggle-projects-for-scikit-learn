import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


data=pd.read_csv('sonar.csv',header=None)
#print(data.head()) #show first five column
#print(data.shape)  # show num of rows and column

#print(data.describe()) #descipe the statical measures of the data

print(data[60].value_counts()) #column 60 which will be R or M we will count num of each value

#Separating data and lables
x=data.drop(columns=60,axis=1)
y=data[60]

#print(x)
#print(y)

#Train an Test data

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.1,random_state=4)

#print(xtrain.shape,xtest.shape)

#the model

model=LogisticRegression()
#train the model
model.fit(xtrain,ytrain)

#Accuracy of training data 

trainPredect=model.predict(xtrain)
score=accuracy_score(trainPredect,ytrain)

print('the accuracy of the training data: ',score)

#Accuracy of test data 

testPredect=model.predict(xtest)
score0=accuracy_score(testPredect,ytest)

print('the accuracy of the test data: ',score0)