import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
import seaborn as sns 
from sklearn.metrics import accuracy_score


data =pd.read_csv('loan data.csv')
#print(data.head())


#num of null values
#print(data.isnull().sum())

#we will delete this null values
data=data.dropna()
#print(data.isnull().sum())
#print(data.head())

#data encodeing :we will replace the textual values by numerical values
data.replace({"Loan_Status":{'N':0,'Y':1},"Married":{'Yes':0,'No':1},
            "Gender":{"Male":0,"Female":1},"Education":{"Graduate":0,"Not Graduate":1},
            "Property_Area":{"Urban":0,"Rural":1,"Semiurban":2},"Self_Employed":{'Yes':0,'No':1}},inplace=True)

#print(data['Dependents'].value_counts())
data.replace({"Dependents":{"3+":3}},inplace=True)
print(data.head())


x=data.drop(columns=["Loan_ID","Loan_Status"],axis=1)
y=data["Loan_Status"]

xtrain,xtest,ytrain,ytest=train_test_split(x,y,random_state=4,test_size=0.2)

model=svm.SVC(kernel="linear")
model.fit(xtrain,ytrain)

#train accuracy
trainpre=model.predict(xtrain)
accu0=accuracy_score(ytrain,trainpre)

print("accuracy of the train is :",accu0)

#test accuracy
testpre=model.predict(xtest)
accu1=accuracy_score(ytest,testpre)

print("accuracy of the train is :",accu1)
