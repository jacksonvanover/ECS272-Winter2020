#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


def concat_embeddings(input_df, embedding1_wt, embedding2_wt, embedding3_wt ):
    weighted1 = []
    weighted2 = []
    weighted3 = []
    
    for item in input_df.word2vec_embeddings.tolist():
        weighted1.append(list(map(lambda x: x*embedding1_wt, item)))
    #print(input_df.embedding1.tolist())
    #print(weighted1)
    for item in input_df.represented_features_embeddings.tolist():
        weighted2.append(list(map(lambda x: x*embedding2_wt, item)))
    #print(input_df.embedding2.tolist())
    #print(weighted2)
    for item in input_df.lbp_embedding.tolist():
        weighted3.append(list(map(lambda x: x*embedding3_wt, item)))
    
    for i in range(len(weighted1)):
        weighted1[i] += weighted2[i] + weighted3[i]         
    
    #print(weighted1)
    
    input_df['concat_embedding'] = weighted1
