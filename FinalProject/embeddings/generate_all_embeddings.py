from pixel_embedding import generate_pixel_embeddings
from represented_features_embedding import generate_represented_features_embeddings
from word_to_vec import generate_word2vec_embeddings

def generate_all_embeddings(df):
    generate_pixel_embeddings(df)
    generate_represented_features_embeddings(df)
    generate_word2vec_embeddings(df)