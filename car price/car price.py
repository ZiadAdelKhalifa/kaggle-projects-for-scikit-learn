import pandas as pd 
import numpy as np
import  matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn import metrics

data=pd.read_csv('car data.csv')
#print(data.head(),"\n",data.shape,"\n",data.describe())

#chick missing values
#print(data.isnull().sum())

#now i will check the types of each column and replace all of them with numbers
#print(data['Transmission'].value_counts())
#print(data['Fuel_Type'].value_counts())
#print(data['Seller_Type'].value_counts())
data.replace({'Transmission':{'Manual':0,'Automatic':1},
                'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2},'Seller_Type':{'Dealer':0,'Individual':1}},inplace=True)

x=data.drop(columns=['Selling_Price','Car_Name'],axis=1)
y=data['Selling_Price']
print(x)
print(y)

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.1,random_state=2)

model0=LinearRegression()
model0.fit(xtrain,ytrain)
#train accuracy with liner regrassion
pretrain=model0.predict(xtrain)
acc0=metrics.r2_score(pretrain,ytrain)
print("r2_score of the train data :",acc0)

#test accuracy with liner regrassion
pretest=model0.predict(xtest)
acc1=metrics.r2_score(pretest,ytest)
print("r2_score of the train data :",acc1)


plt.scatter(ytrain,pretrain)
plt.xlabel("actual price")
plt.ylabel("predicted price")
plt.title("actual price Vs predicted price for train data linear regration")
plt.show()


plt.scatter(ytest,pretest)
plt.xlabel("actual price")
plt.ylabel("predicted price")
plt.title("actual price Vs predicted price for test data linear regration")
plt.show()


#################################
model1=Lasso()
model1.fit(xtrain,ytrain)
#train accuracy with liner regrassion
pretrain=model1.predict(xtrain)
acc00=metrics.r2_score(pretrain,ytrain)
print("r2_score of the train data :",acc00)

#test accuracy with liner regrassion
pretest=model1.predict(xtest)
acc11=metrics.r2_score(pretest,ytest)
print("r2_score of the train data :",acc11)


plt.scatter(ytrain,pretrain)
plt.xlabel("actual price")
plt.ylabel("predicted price")
plt.title("actual price Vs predicted price for train data lasso regration")
plt.show()


plt.scatter(ytest,pretest)
plt.xlabel("actual price")
plt.ylabel("predicted price")
plt.title("actual price Vs predicted price for test data lasso regration")
plt.show()
