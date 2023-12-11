import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder

CaloriesData=pd.read_csv('calories.csv')
ExercisesData=pd.read_csv('exercise.csv')
#print(CaloriesData.head(),'\n',ExercisesData)

data=pd.concat([ExercisesData,CaloriesData['Calories']],axis=1)
# print(data.head())

#print(data.isnull().sum())

#print(data.describe())

encode=LabelEncoder()
data['Gender']=encode.fit_transform(data['Gender'])

"""
#some plots to understand the data

print(data['Gender'].value_counts())
#gender with count num

sns.countplot(x='Gender',data=data)
plt.show()

#ploting the distribution of the 'Age' column
sns.displot(data['Age'])
plt.show()

#ploting the distribution of the 'Height' column
sns.displot(data['Height'])
plt.show()

#ploting the distribution of the 'Height' column
sns.displot(data['Weight'])
plt.show()



#correlation
coree=data.corr()

#constracting the heatmap to understand the correlation
plt.figure(figsize=(10,10))

sns.heatmap(coree,cbar=True,square=True,fmt='.1f',annot=True,cmap='Blues')
plt.show()

"""

x=data.drop(columns=['User_ID','Calories'])
y=data['Calories']

#print(x)
#print(y)

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=2)

model=XGBRegressor()
model.fit(xtrain,ytrain)
#train
train_dataPrediction=model.predict(xtrain)

MAE=metrics.mean_absolute_error(ytrain,train_dataPrediction)

print("mean absoulte error of the train data :",MAE)
#test
test_dataPrediction=model.predict(xtest)

MAE0=metrics.mean_absolute_error(ytest,test_dataPrediction)

print("mean absoulte error of the test data :",MAE0)
