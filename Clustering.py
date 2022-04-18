# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 15:41:12 2022

@author: bhavi
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from kneed import KneeLocator
#import sklearn
#import seaborn as sns
#from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
#from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#from sklearn.cluster import KMeans
from math import sqrt


class Clustering:
    
    def __init__(self,data):
        df=self.data
        print("hello")
    
    def preprocess():
        
        #reading data frame
        df = pd.read_csv(r'C:\Users\bhavi\Downloads\Mall_Customers.csv')
        print(df.head(10))
    
    # updating columns with null values to 0
        df=df.fillna(0)

    # Converting categorical data to numeric
        df['Gender']=df['Gender'].replace(['Male', 'Female'],[0,1])
        print(df.head(10))
        
    def clusters():
        min_range = 2
        max_range = 10

        score1= []
        k_list = range(min_range, max_range)

        for k in k_list:
            km = KMeans(n_clusters = k, random_state= 0)
            km.fit(df) 
            score = km.inertia_
            score1.append(score)

        plt.figure(1 , figsize = (10 ,6))
    # plt.plot(np.arange(min_range , max_range) , inertia , 'o')
        plt.plot(np.arange(min_range , max_range) , score1 , '-' , alpha = 0.5)

        plt.xlabel('Number of Clusters') , plt.ylabel('Inertia score')
        plt.show()
    
    
    def calculate_wcss(data):
        min_range = 2
        max_range = 10
        wcss = []
        for n in range(min_range, max_range):
            kmeans = KMeans(n_clusters=n,random_state=0)
            kmeans.fit(data)
            wcss.append(kmeans.inertia_)
    
        return wcss



    def optimal_number_of_clusters(wcss):
        min_range = 2
        max_range = 10
        x1, y1 = min_range, wcss[0]
        x2, y2 = max_range, wcss[len(wcss)-1]

        distances = []
        for i in range(len(wcss)):
            x0 = i+2
            y0 = wcss[i]
            numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
            denominator = sqrt((y2 - y1)**2 + (x2 - x1)**2)
            distances.append(numerator/denominator)
    
        return distances.index(max(distances)) + 2
    

# calculating the within clusters sum-of-squares for n cluster amounts
    sum_of_squares = calculate_wcss(df)
    
# calculating the optimal number of clusters
    n = optimal_number_of_clusters(sum_of_squares)
    print('Number fo cluster =', n)
    
    number_of_clusters = n


    kmeans = KMeans(n_clusters=n, init ='k-means++', max_iter=300, n_init=10,random_state=0 )
    kmeans.fit(df)

    # Now, print the silhouette score of this model

    print(silhouette_score(df, kmeans.labels_, metric='euclidean'))

    clusters = kmeans.fit_predict(df)
    df["label"] = clusters
     
    fig = plt.figure(figsize=(21,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df.Age[df.label == 0], df["Annual Income (k$)"][df.label == 0], df["Spending Score (1-100)"][df.label == 0], c='blue', s=60)

    ax.scatter(df.Age[df.label == 1], df["Annual Income (k$)"][df.label == 1], df["Spending Score (1-100)"][df.label == 1], c='red', s=60)
    ax.scatter(df.Age[df.label == 2], df["Annual Income (k$)"][df.label == 2], df["Spending Score (1-100)"][df.label == 2], c='green', s=60)
    ax.scatter(df.Age[df.label == 3], df["Annual Income (k$)"][df.label == 3], df["Spending Score (1-100)"][df.label == 3], c='orange', s=60)

    ax.view_init(30, 185)
    plt.show()

    pca = PCA(n_components=4)
    principalComponents = pca.fit_transform(df)

    features = range(pca.n_components_)
    plt.bar(features, pca.explained_variance_ratio_, color='black')
    plt.xlabel('PCA features')
    plt.ylabel('variance %')
    plt.xticks(features)

    PCA_components = pd.DataFrame(principalComponents)

    ks = range(1, 10)
    inertias = []

    for k in ks:
        model = KMeans(n_clusters=k)
        model.fit(PCA_components.iloc[:,:2])
        inertias.append(model.inertia_)

    plt.plot(ks, inertias, '-o', color='black')
    plt.xlabel('number of clusters, k')
    plt.ylabel('inertia')
    plt.xticks(ks)
    plt.show()
    model = KMeans(n_clusters=n)
    model.fit(PCA_components.iloc[:,:2])


    # map back clusters to dataframe

    pred = model.predict(PCA_components.iloc[:,:2])
    frame = pd.DataFrame(df)
    frame['cluster'] = pred
    frame.head()

    avg_df = df.groupby(['cluster'], as_index=False).mean()


    

    