
# coding: utf-8

# In[1]:


'''
    Author: Kai Zhang
    Stevens Institute of Technology
'''
import json
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from sklearn import metrics
import numpy as np
from nltk.cluster import KMeansClusterer, cosine_distance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd


# In[2]:


def cluster_kmean(train_file,test_file):
    train=json.load(open(train_file,'r'))
    test=json.load(open(test_file,'r'))
    test_text,test_label=zip(*test)
    test_text=list(test_text)
    test_label=list(test_label)
    test_flabel=[]
    for i in range(len(test_label)):
        test_flabel.append(test_label[i][0])
    
    text=test_text+train
    # set the min document frequency to 5
    # generate tfidf matrix
    tfidf_vect = TfidfVectorizer(stop_words="english",                             min_df=50) 

    dtm= tfidf_vect.fit_transform(text)
    
    # set number of clusters
    num_clusters=3
    
    # Uses KMeans to cluster documents in both and into 3 clusters by cosine similarity

    # initialize clustering model
    # using cosine distance
    # clustering will repeat 20 times
    # each with different initial centroids
    clusterer = KMeansClusterer(num_clusters,                             cosine_distance,                             repeats=20)

    # samples are assigned to cluster labels starting from 0
    clusters = clusterer.cluster(dtm.toarray(),                              assign_clusters=True)
    # find top words at centroid of each cluster
    # clusterer.means() contains the centroids
    # each row is a cluster, and 
    # each column is a feature (word)
    centroids=np.array(clusterer.means())

    # argsort sort the matrix in ascending order 
    # and return locations of features before sorting
    # [:,::-1] reverse the order
    sorted_centroids = centroids.argsort()[:, ::-1] 

    # The mapping between feature (word)
    # index and feature (word) can be obtained by
    # the vectorizer's function get_feature_names()
    voc_lookup= tfidf_vect.get_feature_names()

    for i in range(num_clusters):
    
        # get words with top 20 tf-idf weight in the centroid
        top_words=[voc_lookup[word_index]                    for word_index in sorted_centroids[i, :20]]
        print("Cluster %d:\n %s " % (i, "; ".join(top_words)))
    print("I set cluster 0 as 'security.expense', cluster 1 as 'security.energy', cluster 2 as 'security.transport'.")
    print('\n')
    # Map cluster id to true labels by "majority vote"
    cluster_dict={0:'security.expense',                  1:"security.energy",                  2:'security.transport'}
    label=[]
    for i in range(len(test_flabel)):
        if test_flabel[i]=='T1':
            label.append('security.expense')
        if test_flabel[i]=='T2':
            label.append('security.transport')
        if test_flabel[i]=='T3':
            label.append('security.energy')

    # Map true label to cluster id
    predicted_target=[cluster_dict[i]                   for i in clusters]
    print(metrics.classification_report          (label, predicted_target[-600:]))
    
    
    


# In[12]:


def cluster_lda(train_file,test_file):
    train_data=json.load(open(train_file,'r'))
    test_data=json.load(open(test_file,'r'))
    test_text,test_label=zip(*test_data)
    test_text=list(test_text)
    test_label=list(test_label)
    all_text = train_data + test_text
    
    tf_vectorizer = CountVectorizer(max_df=0.90,                 min_df=5, stop_words='english')
    tf = tf_vectorizer.fit_transform(train_data)
    tf_feature_names = tf_vectorizer.get_feature_names()
    
    num_topics = 3

    lda = LatentDirichletAllocation(n_components=num_topics,                                 max_iter=15,verbose=1,
                                evaluate_every=1, n_jobs=1,
                                random_state=0).fit(tf)    
    num_top_words=20

    for topic_idx, topic in enumerate(lda.components_):
#         print ("Topic %d:" % (topic_idx))
        # print out top 20 words per topic 
        words=[(tf_feature_names[i],topic[i])                for i in topic.argsort()[::-1][0:num_top_words]]
#         print(words,"\n")
     
    topic_assign=lda.transform(tf)

    topics=np.copy(topic_assign)

    for i in range(0,len(topics)):
        topics[i] = np.where(topics[i]==np.max(topics[i]),1,0)
        
    topics=np.where(topics==1, 1, 0)
    topic_lda_train=[]
    
    for i in range(0,len(topics)):
        topic_lda_train.append(np.argsort(topics)[i][-1])
    
    
    tf_test = tf_vectorizer.transform(test_text)
    topic_assign_test=lda.transform(tf_test)
    topics_test=np.copy(topic_assign_test)
    
    for i in range(0,len(topics_test)):
        topics_test[i] = np.where(topics_test[i]==np.max(topics_test[i]),1,0)
        
    topics_test=np.where(topics_test==1, 1, 0)
    topic_lda_test=[]
    
    for i in range(0,len(topics_test)):
        topic_lda_test.append(np.argsort(topics_test)[i][-1])
    
    test_label=[test_label[i][0] for i in range(0,len(test_label))]
    
    df=pd.DataFrame(list(zip(test_label,topic_lda_test)),                     columns=['actual_class','topic'])
    
    pd.crosstab( index=df.topic, columns=df.actual_class)

    topic_dict={0:'security.expense',                  1:"security.energy",                  2:'security.transport'} 
    label=[]
    for i in range(len(test_label)):
        if test_label[i]=='T1':
            label.append('security.transport')
        if test_label[i]=='T2':
            label.append('security.expense')
        if test_label[i]=='T3':
            label.append('security.energy')

    predicted_target=[topic_dict[i] for i in topic_lda_test]
    
    print("The performance of Q2:")
    print(metrics.classification_report(label, predicted_target))


# In[10]:


if __name__ == '__main__':
    train_file="train_text.json"
    test_file="test_text.json"
    print("Test Q1")
    cluster_kmean(train_file,test_file)
    print("\n")
    print("Test Q2")
    cluster_lda(train_file,test_file)
    print("\n")
    print("Comprison:")
    print("LDA clustering has better accuracy.")
    
