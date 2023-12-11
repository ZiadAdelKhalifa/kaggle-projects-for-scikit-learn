import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

mail_data=pd.read_csv('mail_data.csv')
#print(mail_data.head())

#replacthing the null values with null string
mail_data=mail_data.where(pd.notnull(mail_data),'')

#print(mail_data.head())
#print(mail_data.shape)

#encoding the categorical column with loc function

mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0  # spam = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1  # ham = 1
#separating the data to text and lable
x=mail_data['Message']
y=mail_data['Category']

xtrain,xtest,ytrain,ytest=train_test_split(x,y,random_state=2,test_size=0.2)
print(x.shape)
print(xtrain.shape)
print(xtest.shape)


#transfer the text data to feature vectore that can be input to logistic regrassion
feature_extraction=TfidfVectorizer(min_df=1,stop_words='english')
xtrain=feature_extraction.fit_transform(xtrain)
xtest=feature_extraction.transform(xtest)

#convert ytrain and ytest to integer values
ytrain=ytrain.astype('int')
ytest=ytest.astype('int')

print(xtrain)

#model

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

