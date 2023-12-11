import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


data=pd.read_csv('train.csv')

print(data.head(),"\n",data.shape)

#print(data.isnull().sum())

#we will drop Cabin column and get the mean of the Age column and we have two values in Embarked we will get with mode()
data=data.drop(columns=['Cabin'])
data['Age'].fillna(data['Age'].mean(),inplace=True)
#print(data['Embarked'].mode()[0])
data['Embarked'].fillna(data['Embarked'].mode()[0],inplace=True)
#print(data.isnull().sum())


#print(data.describe())


"""
#num of people survived and not servived
print(data['Survived'].value_counts())
sns.set()
#plot it
sns.countplot(x='Survived',data=data)
plt.show()

#num of each sex
print(data['Sex'].value_counts())
sns.set()
#plot it
sns.countplot(x='Sex',data=data)
plt.show()

#num of the Survivers related with genders
sns.countplot(x='Sex',hue='Survived',data=data)
plt.show()

##
print(data['Pclass'].value_counts())
sns.set()
#plot it
sns.countplot(x='Pclass',data=data)
plt.show()

#num of the Survivers related with genders
sns.countplot(x='Pclass',hue='Survived',data=data)
plt.show()
"""

encode=LabelEncoder()

data['Embarked']=encode.fit_transform(data['Embarked'])
data['Sex']=encode.fit_transform(data['Embarked'])
print(data.head())

x=data.drop(columns=['PassengerId','Name','Ticket','Survived'])
y=data['Survived']

#print(x)
#print(y)

#training and testing data
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
