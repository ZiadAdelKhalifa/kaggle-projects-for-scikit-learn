import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as  sn
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost import XGBRegressor 



"""
we have a problem in the data 
"""

house_price=pd.read_csv('data.csv')
#print(house_price.head())

#print(house_price.describe())
x=house_price.drop(columns=['price','street','city','statezip','country','date'])
y=house_price['price']

#print(x)
#print(y)

#training and spliting the data 

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=2)

#print(xtrain.shape,xtest.shape,ytrain.shape,ytest.shape)


model=XGBRegressor()
model=model.fit(xtrain,ytrain)


#Accuracy of the train 

trainnned=model.predict(xtrain)


#r scored error
score1=metrics.r2_score(xtest,trainnned)
#mean absolute error
score2=metrics.mean_absolute_error(xtest,trainnned)

print("r score error is :" ,score1)
print("Mean absolute error :",score2)

#Accuracy of the test 

testeddd=model.predict(xtest)

#r scored error
score11=metrics.r2_score(ytest,testeddd)
#mean absolute error
score22=metrics.mean_absolute_error(ytest,testeddd)

print("test r score error is :" ,score11)
print("test Mean absolute error :",score22)
