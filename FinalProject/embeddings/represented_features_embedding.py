import pandas as pd

'''
In: the dataFrame containing the entire dataset of chart images

Out: the dataFrame containing then entire dataset of chart images with
a new column of "represented_features_embeddings"

For each row of the dataFrame, generates a one-hot vector encoding
which features from the list DATASET_FEATURES are represented in the
corresponding chart and saves this as a "represented_feature_embedding"
'''

# TODO: update this with the whole list of features when available
DATASET_FEATURES = ["G3", "paid", "higher", "traveltime", "address", "health", "absences", "st_time", "Dalc", "romantic", "internet", "G2", "G1", "sex", "age", "paid", "Walc", "studytime", "Pstatus"]
    
def generate_represented_features_embeddings(df):

    # create dataFrame with an accessible "index" column for reporting
    # ValueErrors in the __which_features function
    df_temp = df.reset_index()

    df["represented_features_embeddings"] = df_temp.apply(lambda x : __which_features(x["index"], x["features"]), axis=1)
    
    return df

def __which_features(index, dim_list):

    represented_features_embedding = [0 for x in range(len(DATASET_FEATURES))]

    for dim in dim_list:
        try:
            represented_features_embedding[DATASET_FEATURES.index(str(dim))] = 1
        except ValueError:
            print("Unrecognized column name in features list of row {}: {}".format(index, dim))

    return represented_features_embedding