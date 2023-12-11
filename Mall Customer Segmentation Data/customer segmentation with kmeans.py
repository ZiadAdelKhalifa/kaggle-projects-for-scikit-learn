import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

data=pd.read_csv('Mall_Customers.csv')

#print(data.head(),'\n',data.shape)

#print(data.isnull().sum())


#we will use two column to plot and know the suitable number for the clusters

x=data.iloc[:,[3,4]].values

#WCSS:within clusters sum of squares

"""
#finding WCSS for diffrent number of clusters
wcss=[]

for i in range(1,12):
    model=KMeans(n_clusters=i,init='k-means++',random_state=42)
    model.fit(data.iloc[:,[3,4]].values)
    wcss.append(model.inertia_)

#ploting the result
sns.set()
plt.plot(range(1,12),wcss)
plt.title("the elbow point graph")
plt.xlabel('num of clusters')
plt.ylabel('WCSS')
plt.show()
"""
#so the suitable num will be 5

kmeans=KMeans(n_clusters=5,init='k-means++',random_state=42)
y=kmeans.fit_predict(x)

print(y)

#plot all clusters and its centroid

plt.figure(figsize=(8,8))

plt.scatter(x[y==0,0],x[y==0,1],c='green',s=50,label='Cluster 1')
plt.scatter(x[y==1,0],x[y==1,1],c='blue',s=50,label='Cluster 2')
plt.scatter(x[y==2,0],x[y==2,1],c='violet',s=50,label='Cluster 3')
plt.scatter(x[y==3,0],x[y==3,1],c='yellow',s=50,label='Cluster 4')
plt.scatter(x[y==4,0],x[y==4,1],c='red',s=50,label='Cluster 5')

#plotting the centroid

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c='cyan',label='centroid')
plt.title('Custmer groups')
plt.xlabel('anaual income')
plt.ylabel('spending score')
plt.show()