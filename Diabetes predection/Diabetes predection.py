import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

data=pd.read_csv('diabetes.csv')
print(data.head())
print(data.shape)

print(data['Outcome'].value_counts())

#Separating data
x=data.drop(columns='Outcome',axis=1) #axis equall 1 means you will drop a column
y=data['Outcome']

#print(x)
#print(y)


scaler=StandardScaler()
new_x=scaler.fit_transform(x)
#print(new_x)

#training and testing data
xtrain,xtest,ytrain,ytest=train_test_split(new_x,y,test_size=0.15,random_state=2)

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


