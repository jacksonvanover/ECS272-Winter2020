import pandas as pd
from sklearn.decomposition import PCA

def pca_columns(df):
    
    #get embedding column, convert to frame
    embed_col = df['concat_embedding']
    embed_frame = []
    for row in embed_col:
        embed_frame.append(row)
    embed_frame = pd.DataFrame(embed_frame)
    
    #do pca, to reduce to 2 axes
    reduce_dim_data_df = pd.DataFrame(PCA(n_components=2).fit_transform(embed_frame))
    
    #rename columns
    dims = ['PC1','PC2']
    for i in range(2):
        reduce_dim_data_df.rename(columns = {i: dims[i]},inplace = True)
    
    #if columns are already there, then replace them
    if 'PC1' in df:
        df['PC1'] = reduce_dim_data_df['PC1']
        df['PC2'] = reduce_dim_data_df['PC2']
        
    #else concatentate PCA values to df
    else:
        df = pd.concat([df,reduce_dim_data_df],axis = 1, sort = False)
    return df
