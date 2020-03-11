from gensim.models import Word2Vec
import numpy as np

def vectorizer (sent, m):
    vec = []
    numw = 0
    for w in sent:
        try:
            if numw == 0:
                vec = m[w]
            else:
    
                vec = np.add(vec, m[w])
            numw += 1
        except:
            pass
    
        return np.asarray(vec) / numw
        

def generate_word2vec_embeddings(data_pick):
        
    m = Word2Vec(data_pick.keywords, size=50, min_count=1 ,sg=1)

    l=[]
    for i in data_pick.keywords:
        l.append(vectorizer(i,m))
        
    X = ''    
    X=np.array(l)   

    l2 = [[] if l[i] is None else l[i].tolist() for i in range(len(l))]

    length = max(map(len, l2))
    y=np.array([xi+[0]*(length-len(xi)) for xi in l2])

    y_mean = np.mean(y)
    y_std = np.std(y)
    y_normal = (y-y_mean)/y_std
    y_normal.shape

    #y_normal
    y_norm_list = [list(y_normal[i]) for i in range(len(y_normal))]
    data_pick['word2vec_embeddings'] = y_norm_list 