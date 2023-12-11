import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor



data=pd.read_csv('gld_price_data.csv')
#print(data.head())
#print(data.shape)
#print(data.describe())


#finding null data
#print(data.isnull().sum())

#making correlation
correlation=data.drop(columns=['Date']).corr()

#making a heat map to understand the correlation
"""
plt.figure(figsize=(8,8))
sns.heatmap(correlation,fmt='0.1f',cmap='Blues',square=True,annot=True)
plt.show()
"""
#corellation values for GLD
"""
print(correlation['GLD'])

sns.displot(data['GLD'],color='green')
plt.show()
"""
x=data.drop(columns=['Date','GLD'])
y=data['GLD']

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=2)

model0=RandomForestRegressor(n_estimators=100)
model0.fit(xtrain,ytrain)

#train accuracy 
pretrain=model0.predict(xtrain)
acc0=metrics.r2_score(pretrain,ytrain)
print("r2_score of the train data :",acc0)

#test accuracy 
pretest=model0.predict(xtest)
acc1=metrics.r2_score(pretest,ytest)
print("r2_score of the train data :",acc1)

#ploting test data
ytest=list(ytest)
plt.plot(ytest,color='blue',label='Actual values')
#plt.plot(pretest,color='green',label='predicted values')
plt.title("actual data Vs predicted data")
plt.xlabel("Number of values")
plt.ylabel("GLD price")
plt.show()