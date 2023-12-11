#problem to solve we didn't use the column of the [Outlet_Size] we want to full the null values and encode it

import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder

data=pd.read_csv('Train.csv')
#print(data.head(),"\n",data.shape,data.describe(),"\n")

#print(data.isnull().sum()) 
#Item_Weight(float value) :we have 1463 null value and Outlet_Size(categorical type with string) : 2410 null value

#we will fill the null values with mean of the item_weight
data['Item_Weight'].fillna(data['Item_Weight'].mean(),inplace=True)
#print(data.isnull().sum())

#replacting the missing values of the Outlet_Size with respect to Outlet_Size
outlet_size=data.pivot_table(values='Outlet_Size',columns='Outlet_Type',aggfunc=(lambda x: x.mode()[0]))
#print(outlet_size)
missing_values=data['Outlet_Size'].isnull()
#print(missing_values)
data.loc[missing_values,'Outlet_Size']=data.loc[missing_values,'Outlet_Type'].apply(lambda x: outlet_size)
#print(data.isnull().sum())
#print(data['Outlet_Size'].isnull().sum())
#print(data['Outlet_Size'].value_counts())
#print(data['Outlet_Size'].value_counts())
#print(data['Item_Weight'].value_counts())

#some plots
"""
plt.figure(figsize=(6,6))
sns.displot(data['Item_Weight'])
plt.show()

plt.figure(figsize=(6,6))
sns.displot(data['Item_Visibility'])
plt.show()

plt.figure(figsize=(6,6))
sns.displot(data['Item_MRP'])
plt.show()

plt.figure(figsize=(6,6))
sns.displot(data['Item_Outlet_Sales'])
plt.show()

plt.figure(figsize=(6,6))
sns.countplot(x='Outlet_Establishment_Year',data=data)
plt.show()

plt.figure(figsize=(6,6))
sns.countplot(x='Item_Fat_Content',data=data)
plt.show()

"""

#print(data['Item_Fat_Content'].value_counts())
#so we had similar data with diffrent names we will sum the same values
data.replace({'Item_Fat_Content':{'low fat':'Low Fat','LF':'Low Fat','reg':'Regular'}},inplace=True)
#data.replace({'Outlet_Size':{'Medium':0,'[Grocery Store]':1,'Small':2,'High':2}},inplace=True)
#print(data['Item_Fat_Content'].value_counts())
#we use encoder insted of replace function
#print(data['Outlet_Size'].value_counts())

encode=LabelEncoder()

data['Item_Fat_Content']=encode.fit_transform(data['Item_Fat_Content'])
data['Item_Type']=encode.fit_transform(data['Item_Type'])
#data['Outlet_Size']=encode.fit_transform(data['Outlet_Size'])
data['Outlet_Location_Type']=encode.fit_transform(data['Outlet_Location_Type'])
data['Item_Identifier']=encode.fit_transform(data['Item_Identifier'])
data['Outlet_Identifier']=encode.fit_transform(data['Outlet_Identifier'])
data['Outlet_Type']=encode.fit_transform(data['Outlet_Type'])

#print(data.head())

x=data.drop(columns=['Item_Outlet_Sales','Outlet_Size'])
y=data['Item_Outlet_Sales']

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=5)

model0=XGBRegressor()
model0.fit(xtrain,ytrain)

#train accuracy with liner regrassion
pretrain=model0.predict(xtrain)
acc0=metrics.r2_score(pretrain,ytrain)
print("r2_score of the train data :",acc0)

#test accuracy with liner regrassion
pretest=model0.predict(xtest)
acc1=metrics.r2_score(pretest,ytest)
print("r2_score of the test data :",acc1)

