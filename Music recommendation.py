#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[3]:


df=pd.read_csv(r"C:\Users\Harsha\Downloads\ex.csv")
df


# In[4]:


df.isnull().sum()


# In[5]:


df.dropna(inplace=True)


# In[6]:


df.isnull().sum()


# In[7]:


df=df.drop_duplicates()


# In[8]:


df.duplicated().sum()


# In[9]:


df.shape


# In[10]:


df.head()


# In[11]:


df['User-Rating']


# In[12]:


l=[]
for i in df['User-Rating']:
    l.append(i[:3])
l


# In[13]:


df['User-Rating']=l
df


# In[14]:


df['Album/Movie']=df['Album/Movie'].str.replace(' ','')
df['Singer/Artists']=df['Singer/Artists'].str.replace(' ','')
df


# In[15]:


df['Singer/Artists']=df['Singer/Artists'].str.replace(',',' ')
df


# In[16]:


df['tags']=df['Singer/Artists']+' '+df['Genre']+' '+df['Album/Movie']+' '+df['User-Rating']
df['tags'][0]


# In[17]:


new_df=df[['Song-Name','tags']]
new_df


# In[18]:


new_df['tags']=new_df['tags'].apply(lambda x:x.lower())
new_df


# In[19]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=2000)


# In[21]:


vectors=cv.fit_transform(new_df['tags']).toarray()


# In[22]:


vectors.shape


# In[23]:


cv.get_feature_names()


# In[24]:


from sklearn.metrics.pairwise import cosine_similarity
similarity=cosine_similarity(vectors)


# In[25]:


sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])


# In[26]:


new_df.rename(columns={'Song-Name':'title'},inplace=True)


# In[27]:


def recommend(music):
    music_index=new_df[new_df['title']==music].index[0]
    distances=similarity[music_index]
    music_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    for i in music_list:
        print(new_df.iloc[i[0]].title)


# In[32]:


recommend('Apna Time Aayega')


# In[29]:


df.head(50)


# In[30]:


import pickle
pickle.dump(new_df,open('musicrec.pkl','wb'))


# In[31]:


pickle.dump(similarity,open('similarities.pkl','wb'))


# In[ ]:




