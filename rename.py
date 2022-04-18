# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 17:03:54 2022

@author: bhavi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator
import sklearn
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import Clustering as cs
obj= cs.printh()
class Analysis:
    
    df = pd.read_csv(r'C:\Users\bhavi\Downloads\Mall_Customers.csv')
    print(df.head(10))
    
    # updating columns with null values to 0
    df=df.fillna(0)
    
    # Converting categorical data to numeric
    df['Gender']=df['Gender'].replace(['Male', 'Female'],[0,1])
    print(df.head(10))
    
    def feature(self,col1,col2):
        X=self.iloc[:, [col1,col2]].values
        
    
    feature(df,3,4)
    cs.print()
    
