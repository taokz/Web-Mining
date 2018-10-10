# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 14:43:09 2018

@author: Kai Zhang
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt

# Q1
def getData():
    data=[] # variable to hold all book data
    page_url="http://books.toscrape.com"
    # your code here
    page = requests.get(page_url)    # send a get request to the web page
    if page.status_code==200:        
        soup = BeautifulSoup(page.content, 'html.parser')
        divs=soup.select("section div ol li article.product_pod")
        #print(divs)
        for idx, article in enumerate(divs):
            title=None
            rating=None
            price=None
            #get title
            a_title=article.select("h3 a")
            #a_title
            if a_title!=[]:
                title=a_title[0].get_text()
                #print title
            
            #get rating
            rating=article.p["class"][1]
            #print rating
            #print(rating)
        
            # get price
            p_price=article.select("div.product_price p.price_color")
            #p_price
            if p_price!=[]:
                price=p_price[0].get_text()
                #print price
                #print(price)

            data.append((title, rating, price))
            #print((title, rating, price))
    return data 
     #call only one time, return on value(the first value)
#Q2
def plot_data(data):
# fill your code here
    #create a pandas dataframe
    df = pd.DataFrame(data, columns=["title","rating","price"])
    df1=df[['rating','price']]
    df1.price=df1.price.str[1:].astype(float) #convert the string to number
    grouped=df1.groupby('rating') #aggeration by "rating"
    ax=grouped.mean().plot.bar(figsize=(8,4), title="avarage price by rating");
    ax.set(xlabel="rating",ylabel="average price");
    
#test the code
if __name__ == "__main__":
    # Test Q1
    data=getData()
    print(data)
    # Test Q2
    plot_data(data)