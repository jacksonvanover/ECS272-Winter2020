#appending the local binary patterns to the dataframe,
#update embeddings
#normalize word2vec

import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from skimage.feature import local_binary_pattern
from skimage.transform import resize

def lbp_hist(df_img, pts = 8, radius = 1):
    
    #get image as numpy array; resize to have consistent number of pixels across all images
    img = np.array(df_img.convert('L'))
    img = resize(img,(500,500))

    #convert with local binary patterns
    lbp_img = local_binary_pattern(img,pts,radius, method = 'default')
    
    #histogram for feature space
    (hist, bin_edges) = np.histogram(lbp_img.ravel(),bins=20)
    hist = list(hist)
    
    #throw out last bin (242+), because it's such an outlier, in all graph cases
    hist.pop(-1)
    
    #scale, with mean and variance
    hist = list(scale(hist))
    
    return hist

def row_norm(row, w_min, w_max):
    
    #add minimum if less than 0
    if w_min < 0:
        for i in range(len(row)):
            row[i] += abs(w_min)
        w_max += abs(w_min)
    
    #divide by max
    for i in range(len(row)):
        row[i] /= w_max
    
    return row
    

df = pd.read_pickle("df_embeddings.pckl")
#concat_embeddings(df, 1, 1, 1)
#lbp_hist(df['image'][0])

#construct local binary pattern column
lbp_df = df['image'].apply(lbp_hist)

#get max value of lbp histogram bin
lbp_max = -2
lbp_min = 2
for row in lbp_df:
    lbp_max = max( (max(row),lbp_max) )
    lbp_min = min( (min(row),lbp_min) )

#normalize to 0, 1
lbp_df = lbp_df.apply(row_norm,args = (lbp_min,lbp_max))

#add column to df
lbp_df = lbp_df.rename("lbp_embedding")
if "lbp_embedding" in df:
    df.drop(columns = "lbp_embedding", inplace = True)
df = pd.concat([df,lbp_df],axis = 1)

#let's fix word2vec as well
w2v_max = -2
w2v_min = 2
for row in df['word2vec_embeddings']:
    w2v_max = max((max(row),w2v_max))
    w2v_min = min((min(row),w2v_min))

#normalize based on min and max
df['word2vec_embeddings'] = df['word2vec_embeddings'].apply(row_norm,args = (w2v_min,w2v_max))

#to csv to inspect
#df.to_csv("embedded_lbp_data.csv")

#to pickle, to push to github
df.to_pickle("df_embeddings.pckl")
    
