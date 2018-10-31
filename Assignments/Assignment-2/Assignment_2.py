# -*- coding: utf-8 -*-
"""
Q1. Define a function to analyze a numpy array
Q2. Define a function to analyze car dataset using pandas

@author: Kai
"""

import numpy as np
import csv
import pandas as pd


def car_analysis(filepath):
    # add your code
    data=pd.read_csv(filepath,header=0)
    print("Find cars with top 3 mpg among those of origin=1: \n")
    top3mpg=data[data.origin==1].sort_values(by='mpg',ascending=False).iloc[0:3,:].loc[:,['mpg','car']]
    print(top3mpg)
    data["brand"]=data.apply(lambda x:x.car.split()[0],axis=1)
    #print("\n The new column: \n")
    #print(data['Brand'])
    print("\nmean, min, and max mpg values for each of these brands: ford, buick and honda")
    print(data[data['brand'].isin(["ford", "buick", "honda"])].groupby('brand').agg(
        {'mpg': [pd.np.mean, pd.np.max, pd.np.min]}))
    print("\ncross tab to show the average mpg of each brand and each origin value")
    print(pd.crosstab(data['brand'], data['origin'], margins=True, values=data['mpg'], aggfunc=pd.np.average))
    
def analyze_tf(arr, binary=False):
    
    tf_idf=None
    
    # add your code
    if binary==True:
        arr_b=np.where(arr>=1,1,0)
        temp=np.sum(arr_b,axis=1)
        tf=arr_b/temp[:,None]
        df=np.sum(np.where(arr_b>=1,1,0),axis=0)
        N=arr_b.shape[0]
        idf=N/df
        tf_idf=tf*np.log(idf)
    else:
        temp=np.sum(arr,axis=1)
        tf=arr/temp[:,None]
        df=np.sum(np.where(arr>=1,1,0),axis=0)
        N=arr.shape[0]
        idf=N/df
        tf_idf=tf*np.log(idf)
    
    return tf_idf

if __name__ == "__main__":  
    
    # Test Question 1
    arr=np.array([[0,1,0,2,0,1],[1,0,1,1,2,0],[0,0,2,0,0,1]])
    
    print(analyze_tf(arr, binary=False))
    # You should get 
    # [[0.         0.27465307 0.         0.20273255 0.         0.10136628]
     # [0.21972246 0.         0.08109302 0.08109302 0.43944492 0.        ]
      #[0.         0.         0.27031007 0.         0.         0.13515504]]
   
    print(analyze_tf(arr, binary=True))
    # You should get
     #[[0.         0.3662041  0.         0.13515504 0.         0.13515504]
     # [0.27465307 0.         0.10136628 0.10136628 0.27465307 0.        ]
     # [0.         0.         0.20273255 0.         0.         0.20273255]]
    
    # test question 2
    
    car_analysis('cars.csv')
