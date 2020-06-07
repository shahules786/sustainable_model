
from nltk.stem import PorterStemmer , WordNetLemmatizer 
import numpy as np

ps = PorterStemmer() 
lemmatizer = WordNetLemmatizer() 




def get_embedding_matrix(word_index,embedding_path,embedding_dim):
        
    num_words = len(word_index)+1

    embedding_dict={}
    with open(embedding_path,'r') as f:
        for line in f:
            values=line.split()
            word=values[0]
            vectors=np.asarray(values[1:],'float32')
            embedding_dict[word]=vectors
    f.close()

    embedding_matrix = np.zeros((num_words,embedding_dim))

    for word,i in word_index.items():

      if i > num_words:
          continue

      emb_vec=embedding_dict.get(word)

      if emb_vec is  None:
          emb_vec=embedding_dict.get(lemmatizer.lemmatize(word))
      elif emb_vec is None:
          emb_vec=embedding_dict.get(ps.stem(word))

      elif emb_vec is not None:
          embedding_matrix[i]=emb_vec
          
    return embedding_matrix




