import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


data0=pd.read_csv('winequality-red.csv')
#print(data0.head(),data0.shape)


#show if there is a null values
#print(data0.isnull().sum())

#describe the data
#print(data0.describe())

#print(data0.value_counts())

"""
#num of values for each quality
sns.catplot(x='quality',data=data0,kind='count')
plt.show()

#plot volatile acidity vs quality
plot=plt.figure(figsize=(5,5))
sns.barplot(x='quality',y='volatile acidity',data=data0)
plt.show()

#plot citric acid vs quality
plot=plt.figure(figsize=(5,5))
sns.barplot(x='quality',y='citric acid',data=data0)
plt.show()
"""
"""
correlation=data0.corr()
#constracting a heat map to understand the correlationn between the column
plt.figure(figsize=(10,10))
sns.heatmap(correlation,cbar=True,square=True,fmt='.1f',annot=True,cmap='Blues')
plt.show()

"""
x=data0.drop(columns='quality',axis=1)


#in y i will make a threshold which will be 7 if >=7 will be 1 else 0
y=data0['quality'].apply(lambda y:1 if y>=7 else 0)


#train test split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=3)

#random forest model
model=RandomForestClassifier()
model.fit(xtrain,ytrain)

#train accuracy
pretrain=model.predict(xtrain)
acc0=accuracy_score(ytrain,pretrain)
print("accuracy of the train is : ",acc0)

#test accuracy
pretest=model.predict(xtest)
acc1=accuracy_score(ytest,pretest)
print("accuracy of the test is : ",acc1)