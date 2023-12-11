import numpy as np 
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


import nltk
#nltk.download('stopwords')

#print(stopwords.words('english'))

data=pd.read_csv('train text.csv')
#print(data.head())
#print(data.iloc[0])


#print(data.isnull().sum()) #counting the number of missing values in data set

# replacing null value with space

data=data.fillna('')
#margine the data (the auter and the text)
data['content']=data['author'] + ' '+ data['title']

#print(data['content'])

x=data.drop(columns='label', axis=1)
y=data['label']
#print(x)
#print(y)


#we will create a funtction that will modefie the text in the data
#which will reducing the word to its root
#like : actor -->actress -->acting --> act
#

port_stem=PorterStemmer()

# The PorterStemmer is a popular stemming algorithm that reduces words 
# to their base or root form by removing suffixes

def stemming(content):
    stemcon=re.sub('^a-zA-Z',' ',content) # the re.sub() function is used to replace any character that is not a letter (a-z and A-Z) #at the start of each word in the content string
    #with a space
    stemcon=stemcon.lower()
    stemcon=stemcon.split() #This line splits the stemcon string into a list of individual words.
    #The split is done based on whitespace characters
    stemcon=[port_stem.stem(word) for word in stemcon if not word in stopwords.words('english')]
    #will chick each word in the list and will return it if it is not stop word
    
    stemcon=' '.join(stemcon) #return the list into string 
    return stemcon

data['content']=data['content'].apply(stemming)
#print(data['content'])

x=data['content'].values
y=data['label'].values



#convert textual data to numerical data 
vectorizer=TfidfVectorizer()
x=vectorizer.fit_transform(x)

#spiliting the data

xtrain,xtest,ytrain,ytest=train_test_split(x,y,random_state=2,test_size=0.2)

model=LogisticRegression()
model.fit(xtrain,ytrain)


#accuracy of the training data

xtrainpredict=model.predict(xtrain)
acu=accuracy_score(xtrainpredict,ytrain)
print('the accuracy of the trained data :',acu)
#accuracy of the test data

xtestpredict=model.predict(xtest)
acu0=accuracy_score(xtestpredict,ytest)
print('the accuracy of the test data :',acu0)