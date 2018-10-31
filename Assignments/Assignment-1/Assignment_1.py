# -*- coding: utf-8 -*-
"""
Q1. Define a function to analyze the frequency of words in a string
Q2. Define a class to analyze a document
Q3. (Bonus) Segment documents by punctuation

Author: Kai Zhang
"""

import csv
from string import punctuation

def count_token(text):
    token_count={}
    list1=(text.lower()).split() #generate a list stores new string
    list2=[] #generate an empty list as temp list
    for i in list1:
        if i not in list2:
            list2.append(i)
            token_count[i]=1
        else:
            token_count[i]+=1
    return token_count

class Text_Analyzer(object):   
    def __init__(self, text):  
        self.input_string=text
        self.token_count={}
         
    def analyze(self):   
        self.token_count=count_token(self.input_string)
        
    def save_to_file(self, filepath="test.csv"):
        keys=self.token_count.keys() #generate the keys list
        values=self.token_count.values() #generate the values list
        zipped=list(zip(keys, values))
        #print(zipped)
        with open(filepath,"w",newline='') as f:
            writer=csv.writer(f, delimiter=',')
            writer.writerows(zipped)
            # the result will be 16*2(an extra empty line between each token)
            # it seems that this problem only happens in windows
            # resolved: add " newline='' "

def corpus_analyze(docs):
    token_freq={}
    token_to_doc={}
    temp=" ".join(docs) # convert a list of strings into a string
    #print("the result by converting: ")
    #print(temp)
    temp1=[]
    temp2=[]
    temp3=[]
    # add your code
    #newString=''.join(c for c in temp if c not in punctuation)
    #because of wrong retult ----> it's will be its instead of (it s)
    translator = str.maketrans(punctuation, ' '*len(punctuation)) #map punctuation to space
    newString=temp.translate(translator)
    #print("delete the punctuation") 
    #print(newString)
    temp1=(newString.lower()).split() #generate the list without whitespaces
    for a in temp1:
        if len(a)==1:
            temp1.remove(a)
    #print("the new string is :")
    #print(temp1) 
    for i in temp1:
        if i not in temp2:
            temp2.append(i)
            token_freq[i]=1
        else:
            token_freq[i]+=1
    #print(temp2)
    count=-1
    low=[x.lower() for x in docs]
    #print(low)
    for x in temp2:
        for y in low:
            if x in y:
                count+=1
                temp3.append(count)
                token_to_doc[x]=temp3
            else:
                count+=1
    #token_to_doc=0
    
    return token_freq, token_to_doc

# best practice to test your class
# if your script is exported as a module,
# the following part is ignored
# this is equivalent to main() in Java

if __name__ == "__main__":  
    
    # Test Question 1
    text='''Hello world!
        It's is a hello world example !'''   
    print(count_token(text))
    
    # # The output of your text should be: 
    # {'world': 1, '!': 1, 'world!': 1, 'a': 1, "it's": 1, 
    # 'example': 1, 'hello': 2, 'is': 1}
    # # My result: {'hello': 2, 'world!': 1, "it's": 1, 'is': 1, 'a': 1, 
    #'world': 1, 'example': 1, '!': 1}
    
    # Test Question 2
    analyzer=Text_Analyzer(text)
    analyzer.analyze()
    analyzer.save_to_file("C:/Users/yongk/Documents/PythonLearning/test.csv")
    # You should be able to find the csv file with 8 lines, 2 columns
    
    #3 Test Question 3
    docs=['Hello world!', "It's is a hello world example !"]
    word_freq, token_to_doc=corpus_analyze(docs)   
    print(word_freq)
    # output should be {'hello': 2, 'world': 2, 'it': 1, 'is': 1, 'example': 1}
    print(token_to_doc)
    print("The result of token_to_doc is wrong, I should modify the code then")
    # output should be {'hello': [0, 1], 'world': [0, 1], 'it': [1], 'is': [1], 'example': [1]}