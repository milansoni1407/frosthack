# -*- coding: utf-8 -*-
"""
Created on Sat May 21 23:47:12 2022

@author: paart
"""

import pandas as pd
import scipy as sc
import ast
import numpy as np
from nltk.stem.porter import PorterStemmer 
ps = PorterStemmer()
data = pd.read_csv('dataset.csv').fillna('[]')[:10000]
remove = ['dimension-x', 'dimension-y', 'dimension-z', 'edition',
'edition-statement', 'format', 'id', 'illustrations-note','imprint', 'index-date',
'isbn10', 'isbn13', 'publication-date', 'publication-place','weight','for-ages','rating-avg','rating-count','bestsellers-rank']
data = data.drop(remove,axis=1)
author= pd.read_csv('authors.csv').fillna('[]')
category= pd.read_csv('categories.csv').fillna('[]')
def convert(lis):
    l=[]
    for i in ast.literal_eval(lis):
        index= category[category["category_id"]==i]
        l.append(ps.stem(index["category_name"].tolist()[0]))
    return l
def convert2(lis):
    l=[]
    for i in ast.literal_eval(lis):
        index= author[author["author_id"]==i]
        l.append(ps.stem(index["author_name"].tolist()[0]))
    return l
data['categories2'] = ""
data['author2'] = ""
data['title2'] = ""
data['description2'] = ""
for i in range(len(data['categories'])):
    data.at[i,'categories2'] = convert(data.loc[i,'categories'])
    data.at[i,'description2'] = [ps.stem(i) for i in data.loc[i,'description'].split(" ")]
    data.at[i,'title2'] = [ps.stem(i) for i in data.loc[i,'title'].split(" ")]
    data.at[i,'authors2'] = convert2(data.loc[i,'authors'])



# data['categories'] = data['categories'].apply(lambda x:[i.replace(" ","") for i in x])
# data['description'] = data['description'].apply(lambda x:[i.replace(" ","") for i in x])
# data['title'] = data['title'].apply(lambda x:[i.replace(" ","") for i in x])
# data['authors'] = data['authors'].apply(lambda x:[i.replace(" ","") for i in x])
data['tags'] = data['categories2'].apply(lambda x:[i.replace(" ","") for i in x])+data['title2'].apply(lambda x:[i.replace(" ","") for i in x])+data['authors2'].apply(lambda x:[i.replace(" ","") for i in x])+data['description2'].apply(lambda x:[i.replace(" ","") for i in x])
data['tags'] = data['tags'].apply(lambda x: (" ".join(x)).lower())
data.to_csv('clean_data_sample.csv',index=False)

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features = 7000,stop_words = 'english')
vectors = cv.fit_transform(data['tags']).toarray()

from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)

def recommend(book):
    book_index = data[data['title']==book].index[0]
    book_vec = list(enumerate(similarity[book_index]))
    index = sorted(book_vec,reverse=True,key = lambda x:x[1])[1:10][0]
    res = []
    for i in index:
        res.append(data.loc[i]['title'])
        # print(data.iloc[i].title)
    return res
data_dict = data.to_dict()
import pickle
import joblib

pickle.dump(similarity,open('similarity.pkl','wb'))
pickle.dump(data_dict,open('data_dict.pkl','wb'))
# recommendation_data = pd.DataFrame(columns = ['title', 'recommend'])
# recommendation_data['url'] = data['url']
# recommendation_data['image-url'] = data['image-url']
# for i in range(len(data['title'])):
#     recommendation_data['title'][i] = data['title'][i]
#     recommendation_data.at[i,'recommend'] = recommend(data['title'][i])









    
    