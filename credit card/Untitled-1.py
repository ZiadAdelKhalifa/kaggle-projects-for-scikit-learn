import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data=pd.read_csv('creditcard.csv')
#print(data.head())
#print(data.shape)
#print(data.describe())
#print(data.isnull().sum())

#print( data['Class'].value_counts()) #we will find that the data is highly unbalanced

zeroes=data[data.Class ==0]
ones=data[data.Class ==1]
#print(zeroes.shape)
#print(ones.shape)
#print(zeroes.Amount.describe())
#print(data.groupby('Class').mean())
zeroes=zeroes.sample(n=492)  #we reduce num of zeroes to be equal the num of ones

new_data=pd.concat([zeroes,ones],axis=0)
#print(new_data.head(),new_data.shape)
x=new_data.drop(columns=['Class'],axis=1)
y=new_data['Class']


xtrain,xtest,ytrain,ytest=train_test_split(x,y,random_state=3,test_size=0.2)

model=LogisticRegression()
model.fit(xtrain,ytrain)

#accuracy of the training data

xtrainpredict=model.predict(xtrain)
acu=accuracy_score(xtrainpredict,ytrain)
print('the accuracy of the trained data :',acu)
#accuracy of the test data

xtestpredict=model.predict(xtest)
acu0=accuracy_score(xtestpredict,ytest)
print('the accuracy of the test data :',acu0)