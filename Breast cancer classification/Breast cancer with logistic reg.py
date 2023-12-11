import numpy as np 
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data=load_breast_cancer()
x=data.data
y=data.target

#print(len(x))
#print(len(y))

#we can inster teh data in data frame to do some operation on it
data_frame=pd.DataFrame(data.data,columns=data.feature_names)
print(data_frame)
#adding the lable
data_frame['lable']=data.target
#print(data_frame.shape,data_frame.head(),
#    data_frame.describe())

#checking the null values
#print(data_frame.isnull().sum())

xtrain,xtest,ytrain,ytest=train_test_split(x,y,random_state=2,test_size=0.2)

model=LogisticRegression()
model.fit(xtrain,ytrain)


#train accuracy
preTrain=model.predict(xtrain)
acc=accuracy_score(ytrain,preTrain)

print("Accuracy of the train data is :",acc)

#test accuracy
preTest=model.predict(xtest)
acc0=accuracy_score(ytest,preTest)

print("Accuracy of the test data is :",acc0)