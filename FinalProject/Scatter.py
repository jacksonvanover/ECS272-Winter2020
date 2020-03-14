#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objs as go
import numpy as np
import requests
import json
from collections import Counter
import scipy as sp
import os
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import nltk
import codecs
from sklearn import feature_extraction
import mpld3
from nltk.stem.snowball import SnowballStemmer
import re
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords


# In[ ]:


default_clusters = 4 #randomly chose 4
embedding_combo_array = np.array(test_df['concat_embedding'].tolist())
km = KMeans(n_clusters=default_clusters, random_state=10).fit(embedding_combo_array)
image_df['cluster'] = km.labels_ 

fig = px.scatter(image_df, x="PC1", y="PC2", color="cluster",
                 hover_data=['title']) #PC1 and PC2 are the principal components
#fig.show()


# In[ ]:


@app.callback(
        [Output('scatter', 'figure')],
        [Input('wt1_slider', 'value'),
        Input('wt2_slider', 'value'),
        Input('wt3_slider', 'value'),
        Input('cluster_slider', 'value')]
    )
    def update_chart(wt1, wt2, wt3, k):
        concat_embeddings(df, wt1, wt2, wt3)
        embedding_combo_array = np.array(test_df['concat_embedding'].tolist())
        km = KMeans(n_clusters=k, random_state=10).fit(embedding_combo_array)
        image_df['cluster'] = km.labels_ 
        
        #recalculate PCA#
        
        fig = px.scatter(image_df, x="PC1", y="PC2", color="cluster",
                 hover_data=['title'])

