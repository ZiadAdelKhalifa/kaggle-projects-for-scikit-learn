import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics 
data=pd.read_csv('insurance.csv')

#print(data.head(),"\n",data.shape)
#print(data.isnull().sum())

#some plots
"""
#age distribution
plt.figure(figsize=(6,6))
sns.displot(data['age'])
plt.title('Age distribution')
plt.show()

#gender column
plt.figure(figsize=(6,6))
sns.countplot(x='sex',data=data)
plt.title('sex distribution')
plt.show()

#bmi distribution
plt.figure(figsize=(6,6))
sns.displot(data['bmi'])
plt.title('bmi distribution')
plt.show()

#children column
plt.figure(figsize=(6,6))
sns.countplot(x='children',data=data)
plt.title('children distribution')
plt.show()

#smoker column
plt.figure(figsize=(6,6))
sns.countplot(x='smoker',data=data)
plt.title('smoker distribution')
plt.show()

#charges distribution
plt.figure(figsize=(6,6))
sns.displot(data['charges'])
plt.title('charges distribution')
plt.show()

"""

#encodeing the data
#print(data['region'].value_counts())
data.replace({"sex":{'male':0,'female':1},"smoker":{'yes':0,'no':1},"region":{'southeast':0,'southwest':1,'northwest':2,'northeast':3}},inplace=True)
print(data)


x=data.drop(columns=['charges'],axis=1)
y=data['charges']

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=2)


model0=LinearRegression()
model0.fit(xtrain,ytrain)
#train accuracy with liner regrassion
pretrain=model0.predict(xtrain)
acc0=metrics.r2_score(pretrain,ytrain)
print("r2_score of the train data :",acc0)

#test accuracy with liner regrassion
pretest=model0.predict(xtest)
acc1=metrics.r2_score(pretest,ytest)
print("r2_score of the test data :",acc1)
